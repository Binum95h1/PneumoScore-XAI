"""Kaggle inference and verification for the team's two trusted checkpoints."""
from pathlib import Path
import os,io,json,time,hashlib,pickle,zipfile,platform,shutil,gc
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18

VERSION='pneumoscore_ensemble_export_v1_1'
LEGACY_MODEL_SHA256='d8055289142df2493d3f0e573e3e1420d2626ba7e7b64b8d05a14646497c9032'
PREPROCESS_ID='exif_gray_pad_bilinear_rgb_normalize_0.5_v1'
LABEL_NAMES={0:'Normal',1:'Pneumonia'}
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def ensure(condition,message):
    if not condition:raise ValueError(message)

class BuiltinsOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise pickle.UnpicklingError('Globals forbidden in portable model: '+module+'.'+name)

class KnownLegacyUnpickler(pickle.Unpickler):
    # Used only AFTER the exact team's old file SHA256 has matched.
    def find_class(self,module,name):
        if module=='numpy' and name in {'ndarray','dtype'}:return getattr(np,name)
        if module in {'numpy.core.multiarray','numpy._core.multiarray'} and name in {'_reconstruct','scalar'}:
            try:core=np._core.multiarray
            except AttributeError:core=np.core.multiarray
            return getattr(core,name)
        raise pickle.UnpicklingError('Unsupported legacy global: '+module+'.'+name)

def plain_value(value):
    if isinstance(value,np.generic):return plain_value(value.item())
    if value is None or type(value) in {str,bytes,int,float,bool}:return value
    if isinstance(value,(list,tuple)):return [plain_value(x) for x in value]
    if isinstance(value,dict):
        ensure(all(type(k) is str for k in value),'Metadata keys must be strings.')
        return {k:plain_value(v) for k,v in value.items()}
    raise ValueError('Unsupported metadata value: '+type(value).__name__)

def encode_portable(payload,source_sha256=None):
    result=plain_value({k:v for k,v in payload.items() if k!='members'})
    result['export_schema']='pneumoscore_probability_ensemble_bytes_v2'
    result['storage_version']=2
    if source_sha256:result['converted_from_sha256']=source_sha256
    result['members']=[]
    for member in payload['members']:
        tensors={}
        for key,value in member['state_dict_numpy'].items():
            a=np.ascontiguousarray(value) if value.ndim else np.asarray(value).copy()
            ensure(a.dtype.kind in 'fiu' and not a.dtype.hasobject,'Only numeric weight arrays allowed.')
            tensors[key]={'dtype':a.dtype.str,'shape':list(a.shape),'data':a.tobytes(order='C')}
        result['members'].append({'metadata':plain_value(member['metadata']),'state_dict_bytes':tensors})
    return result

def decode_portable(raw):
    ensure(raw.get('export_schema')=='pneumoscore_probability_ensemble_bytes_v2','Unknown portable schema.')
    ensure(type(raw.get('members')) is list and len(raw['members'])==2,'Expected two ensemble members.')
    result={k:v for k,v in raw.items() if k!='members'};result['members']=[]
    for member in raw['members']:
        arrays={}
        for key,item in member['state_dict_bytes'].items():
            dtype=np.dtype(item['dtype']);shape=item['shape'];data=item['data']
            ensure(dtype.kind in 'fiu' and not dtype.hasobject and dtype.fields is None,'Unsupported weight dtype.')
            ensure(type(shape) is list and all(type(x) is int and 0<=x<=100000 for x in shape) and len(shape)<=5,'Invalid weight shape.')
            count=1
            for dimension in shape:count*=dimension
            ensure(type(data) is bytes and len(data)==count*dtype.itemsize,'Invalid weight bytes length.')
            arrays[key]=np.frombuffer(data,dtype=dtype).reshape(tuple(shape)).copy()
        result['members'].append({'metadata':member['metadata'],'state_dict_numpy':arrays})
    return result

def read_model_payload(path):
    path=Path(path);ensure(0<path.stat().st_size<=128*1024**2,'Unexpected model file size.')
    digest=file_hash(path)
    with path.open('rb') as f:
        try:raw=BuiltinsOnlyUnpickler(f).load()
        except pickle.UnpicklingError:
            ensure(digest==LEGACY_MODEL_SHA256,'Legacy NumPy pickle is accepted only for the exact previously verified team file.')
            f.seek(0);raw=KnownLegacyUnpickler(f).load()
    ensure(type(raw) is dict,'Model payload must be a dictionary.')
    if raw.get('export_schema')=='pneumoscore_probability_ensemble_bytes_v2':payload=decode_portable(raw)
    else:
        ensure(digest==LEGACY_MODEL_SHA256 and raw.get('export_schema')=='pneumoscore_probability_ensemble_numpy_v1','Unknown legacy model.')
        payload=raw
    ensure(payload.get('preprocess')==PREPROCESS_ID and payload.get('image_size')==224,'Preprocessing mismatch.')
    ensure(payload.get('weights')==[.5,.5] and payload.get('seed')==2026 and payload.get('threshold')==.55,'The selected ensemble must remain fixed: seed2026, weights50/50, threshold0.55.')
    ensure(payload.get('label_mapping')=={'Normal':0,'Pneumonia':1},'Internal label mapping mismatch.')
    ensure(len(payload.get('members',[]))==2,'Expected two models.')
    return payload

def save_portable_pickle(payload,path,source_sha256=None):
    raw=encode_portable(payload,source_sha256)
    temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f:pickle.dump(raw,f,protocol=4)
    os.replace(temp,path)
    with Path(path).open('rb') as f:check=decode_portable(BuiltinsOnlyUnpickler(f).load())
    for old,new in zip(payload['members'],check['members']):
        ensure(old['state_dict_numpy'].keys()==new['state_dict_numpy'].keys(),'Portable tensor keys changed.')
        for key,a in old['state_dict_numpy'].items():
            b=new['state_dict_numpy'][key]
            ensure(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),('Portable tensor changed',key))
    return check

def execution_environment():
    import PIL
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pillow=PIL.__version__,
        torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,precision='float32',code_version=VERSION)

def require_kaggle(t4=False):
    ensure(Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir(),'Run on Kaggle.')
    if t4:
        ensure(torch.cuda.is_available(),'Select GPU T4.')
        ensure('T4' in torch.cuda.get_device_name(0),'Verification requires T4.')
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def save_json(value,path):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def manifest_fingerprint(frame):
    cols=['relative_path','label','split','sha256','pixel_sha256','patient_id','group_id']
    content=frame.sort_values('relative_path')[cols].to_csv(index=False,lineterminator='\n')
    return hashlib.sha256(content.encode()).hexdigest()

def read_xray(path):
    with Image.open(path) as source:
        if source.mode in {'I','F'} or source.mode.startswith('I;16'):
            raise ValueError('16-bit/float input needs a BTC-approved normalization policy; no automatic intensity conversion.')
        ensure(getattr(source,'n_frames',1)==1,'Multi-frame image needs an explicit BTC-approved frame policy.')
        return ImageOps.exif_transpose(source).convert('L')

class SquarePad:
    def __init__(self,size):self.size=int(size)
    def __call__(self,image):
        return ImageOps.pad(image,(self.size,self.size),method=Image.Resampling.BILINEAR,color=0).convert('RGB')

def inference_transform(size=224):
    return T.Compose([SquarePad(size),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])

class ResNet18Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=resnet18(weights=None)
        self.backbone.fc=nn.Linear(512,1)
    def forward(self,x):
        return self.backbone(x).squeeze(1)

class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('ECA kernel_size must be positive and odd.')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_filter = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
    def forward(self, features):
        descriptor = self.pool(features).flatten(1).unsqueeze(1)
        gate = self.channel_filter(descriptor).squeeze(1).sigmoid()
        return features * gate[:, :, None, None]

class ResNet18ECA(nn.Module):
    def __init__(self, use_eca=True, kernel_size=5):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        # Create ECA after the backbone and classifier, so a matched seed keeps them identical to the baseline.
        self.eca = ECA(kernel_size) if use_eca else nn.Identity()
        self.use_eca = bool(use_eca)
        self.kernel_size = int(kernel_size)
    def forward_features(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        return self.eca(b.layer4(x))
    def forward(self, x):
        features = self.forward_features(x)
        pooled = self.backbone.avgpool(features).flatten(1)
        return self.backbone.fc(pooled).squeeze(1)

def network_from_state(metadata,state):
    arch=metadata['architecture']
    if arch=='resnet18':net=ResNet18Baseline()
    elif arch=='resnet18_eca':net=ResNet18ECA(use_eca=True,kernel_size=int(metadata['eca_kernel']))
    else:raise ValueError('This bundle supports only the chosen baseline/ECA pair.')
    net.load_state_dict(state,strict=True)
    return net

class ProbabilityEnsemble(nn.Module):
    def __init__(self,members,weights):
        super().__init__();self.members=nn.ModuleList(members)
        ensure(len(weights)==len(members)==2 and all(w>0 for w in weights),'Invalid ensemble member weights.')
        ensure(np.isclose(sum(weights),1.),'Weights must sum to1.')
        self.weights=tuple(float(w) for w in weights)
    def forward(self,images):
        # This forward returns probabilities, so do not apply sigmoid a second time.
        result=self.weights[0]*self.members[0](images).sigmoid()
        return result+self.weights[1]*self.members[1](images).sigmoid()

def validate_member(saved,arch,seed,manifest_hash=None):
    expected=dict(architecture=arch,seed=int(seed),pretrained=False,initialization='scratch',
        image_size=224,input_channels=3,label_mapping={'Normal':0,'Pneumonia':1},preprocess=PREPROCESS_ID)
    if manifest_hash:expected['manifest_hash']=manifest_hash
    if arch=='resnet18_eca':expected['eca_kernel']=5
    for key,value in expected.items():ensure(saved.get(key)==value,('Checkpoint metadata mismatch',key))
    ensure(isinstance(saved.get('state_dict'),dict) and saved.get('phase')=='main','Invalid state/phase.')
    return saved

def find_run(arch,seed):
    found={}
    for p in Path('/kaggle/input').rglob('run_config.json'):
        if not (p.parent/'best.pt').is_file():continue
        cfg=json.loads(p.read_text())
        if cfg.get('ARCH')==arch and int(cfg.get('SEED',-1))==seed:found[str(p.parent.resolve())]=p.parent
    assert len(found)==1,('Attach the Stability notebook Output; expected one run',arch,seed,len(found))
    return next(iter(found.values()))

def find_selection(seed):
    choices=list(Path('/kaggle/input').rglob('selected_candidate.json'))
    assert len(choices)==1,'Attach the saved Ensemble Error Analysis notebook Output.'
    chosen=json.loads(choices[0].read_text())
    assert chosen['architectures']==['resnet18','resnet18_eca'] and chosen['weights']==[.5,.5]
    threshold=float(chosen['per_seed_tuned_thresholds'][str(seed)])
    assert 0<threshold<1
    return chosen,threshold,str(choices[0])

def resolve_data_root(config,manifest):
    first=Path(str(manifest.iloc[0].relative_path))
    assert not first.is_absolute() and '..' not in first.parts
    configured=Path(config['DATA_ROOT']) if config.get('DATA_ROOT') else None
    if configured and (configured/first).is_file():return configured
    found={}
    for p in Path('/kaggle/input').rglob(first.name):
        candidate=p
        for _ in first.parts:candidate=candidate.parent
        if (candidate/first).is_file():found[str(candidate.resolve())]=candidate
    assert len(found)==1,'Attach the BTC dataset; set DATA_ROOT if more than one copy is attached.'
    return next(iter(found.values()))

class InferenceDataset(Dataset):
    def __init__(self,frame,root=None):
        self.frame=frame.reset_index(drop=True).copy();self.root=root
        self.transform=inference_transform()
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        r=self.frame.iloc[index]
        path=Path(r.path) if self.root is None else self.root/str(r.relative_path)
        image_id=str(r.Image_id) if self.root is None else str(r.image_id)
        return self.transform(read_xray(path)),image_id

def loader_for(frame,config,root=None):
    return DataLoader(InferenceDataset(frame,root),batch_size=int(config['BATCH_SIZE']),shuffle=False,
                      num_workers=int(config['NUM_WORKERS']),pin_memory=torch.cuda.is_available(),
                      persistent_workers=False,generator=torch.Generator().manual_seed(2027))

@torch.inference_mode()
def predict(net,loader,device,probability_output=False):
    net.eval();rows=[]
    if device.type=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for images,ids in loader:
        value=net(images.to(device,non_blocking=True))
        p=value if probability_output else value.sigmoid()
        ensure(p.ndim==1 and torch.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid predicted probabilities.')
        rows.extend({'image_id':str(i),'p_pneumonia':float(x)} for i,x in zip(ids,p.cpu().tolist()))
    if device.type=='cuda':torch.cuda.synchronize()
    return pd.DataFrame(rows),time.perf_counter()-start

def metrics(y,p,threshold):
    pred=np.asarray(p)>=threshold;y=np.asarray(y,dtype=int)
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    a=(tn+tp)/len(y);se=tp/(tp+fn);sp=tn/(tn+fp);f1=2*tp/(2*tp+fp+fn)
    return dict(threshold=float(threshold),Accuracy=a,Sensitivity=se,Specificity=sp,F1=f1,
                P=.30*a+.25*se+.20*sp+.25*f1,TN=tn,FP=fp,FN=fn,TP=tp)

def compare_probabilities(actual,reference,threshold):
    a=actual.sort_values('image_id').reset_index(drop=True)
    b=reference.sort_values('image_id').reset_index(drop=True)
    assert a.image_id.is_unique and a.image_id.tolist()==b.image_id.tolist(),'Image IDs changed.'
    aa=a.p_pneumonia.to_numpy();bb=b.p_pneumonia.to_numpy()
    assert np.allclose(aa,bb,atol=1e-6,rtol=1e-5),'Probability reproduction failed.'
    assert np.array_equal(aa>=threshold,bb>=threshold),'Prediction labels changed.'
    return dict(n_images=len(a),max_absolute_probability_error=float(np.max(np.abs(aa-bb))),
                changed_labels=int(((aa>=threshold)!=(bb>=threshold)).sum()),atol=1e-6,rtol=1e-5)

def export_pickle(saved_members,weights,threshold,manifest,path,source_records):
    keys=['schema','code_version','architecture','eca_kernel','seed','pretrained','initialization',
          'image_size','input_channels','label_mapping','preprocess','manifest_hash','initial_backbone_hash',
          'phase','best_epoch','best_score','best_loss','threshold']
    payload=dict(export_schema='pneumoscore_probability_ensemble_numpy_v1',version=VERSION,
        architecture='resnet18_plus_resnet18_eca_probability_ensemble',seed=2026,weights=list(weights),
        threshold=float(threshold),threshold_origin='locked_existing_validation_selection_no_retuning',
        preprocess=PREPROCESS_ID,image_size=224,label_mapping={'Normal':0,'Pneumonia':1},
        manifest_hash=saved_members[0]['manifest_hash'],source_checkpoints=source_records,
        seen_hashes={'file_sha256':sorted(set(manifest.sha256)),'pixel_sha256':sorted(set(manifest.pixel_sha256))},
        members=[dict(metadata={k:s[k] for k in keys if k in s},
          state_dict_numpy={k:v.detach().cpu().contiguous().numpy().copy() for k,v in s['state_dict'].items()}) for s in saved_members])
    save_portable_pickle(payload,path)
    return payload

def load_pickle(path,device):
    payload=read_model_payload(path);nets=[]
    for item,arch in zip(payload['members'],['resnet18','resnet18_eca']):
        state={k:torch.from_numpy(v.copy()) for k,v in item['state_dict_numpy'].items()}
        meta=dict(item['metadata'],state_dict=state)
        validate_member(meta,arch,payload['seed'],payload['manifest_hash'])
        nets.append(network_from_state(meta,state))
    ensure(nets[0].state_dict().keys()!=nets[1].state_dict().keys(),'Wrong ensemble architectures.')
    return ProbabilityEnsemble(nets,payload['weights']).to(device).eval(),payload

@torch.inference_mode()
def benchmark_forward(nets,images,device):
    records=[]
    for batch in [1,len(images)]:
        x=images[:batch].to(device)
        for name,net in nets.items():
            for _ in range(10):net(x)
            torch.cuda.synchronize();times=[]
            for _ in range(50):
                begin=time.perf_counter();net(x);torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-begin))
            records.append(dict(model=name,batch_size=batch,warmup_batches=10,repetitions=50,
                median_batch_ms=float(np.median(times)),p95_batch_ms=float(np.percentile(times,95)),
                amortized_ms_per_image=float(np.median(times)/batch),
                throughput_images_per_second=float(1000*batch/np.median(times)),
                scope='FP32 forward only, input already on GPU, synchronous wall time; not disk/preprocessing'))
    return records

def validate_csv(frame,path):
    ensure(list(frame.columns)==['Image_id','Prediction'] and len(frame)>0,'CSV must contain Image_id,Prediction and at least one row.')
    ensure(frame.Image_id.is_unique and frame.Image_id.map(lambda x:isinstance(x,str) and bool(x.strip())).all(),'CSV IDs must be unique nonempty strings.')
    ensure(set(frame.Prediction).issubset({'Normal','Pneumonia'}),'BTC Article6 requires Normal/Pneumonia strings, not internal0/1.')
    frame.to_csv(path,index=False)
    loaded=pd.read_csv(path,dtype=str,keep_default_na=False)
    ensure(loaded.equals(frame),'CSV round-trip mismatch.')

def make_submission_notebook(module_source):
    config=dict(MODE='submit',MODEL_PATH='',TEST_DIR='',SAMPLE_SUBMISSION_CSV='',TEAM_ID='team01',
                BATCH_SIZE=32,NUM_WORKERS=2,OUTPUT='/kaggle/working/ensemble_submission')
    def code(s):return dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[])
    return dict(nbformat=4,nbformat_minor=4,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'},'kaggle':{'isInternetEnabled':False,'isGpuEnabled':True}},cells=[
        dict(cell_type='markdown',metadata={},source='# PneumoScore ensemble — suy luận Test\nAdd Input model.pkl của đội và tập Test BTC. Điền TEST_DIR, mã đội và CSV mẫu nếu BTC cung cấp, rồi Run All trên Kaggle. Không cần nhãn Test, không huấn luyện lại, không sửa CSV thủ công. Trọng số ghép và ngưỡng nằm trong model.pkl.\n'),
        code('CONFIG='+repr(config)+'\n'),code(module_source),
        code("CONFIG['_MODULE_SOURCE']="+repr(module_source)+"\nrun_submission(CONFIG)\n")])

def make_verification_notebook(config,module_source):
    def code(s):return dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[])
    return dict(nbformat=4,nbformat_minor=4,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'},'kaggle':{'isInternetEnabled':False,'isGpuEnabled':True}},cells=[
        dict(cell_type='markdown',metadata={},source='# Kiểm tra lại ensemble seed 2026\nChạy trên Kaggle GPU T4. Add Input Output Stability 3 Seeds, Output Ensemble Error Analysis 3 Seeds và Dataset_phổi. Không train lại, không tải pretrained.\n'),
        code('CONFIG='+repr(config)+'\n'),code(module_source),
        code('MODULE_SOURCE='+repr(module_source)+'\nverify_export(CONFIG,MODULE_SOURCE)\n')])

def resolve_model(config):
    if config.get('MODEL_PATH'):
        p=Path(config['MODEL_PATH']);ensure(p.is_file(),'MODEL_PATH does not exist.');return p
    choices={str(p.resolve()):p for p in Path('/kaggle/input').rglob('model.pkl')}
    ensure(len(choices)==1,'Set MODEL_PATH explicitly if zero or multiple model.pkl files are attached.')
    return next(iter(choices.values()))

def collect_test(config,payload,out):
    ensure(config.get('TEST_DIR'),'Set TEST_DIR to BTC Test images.')
    root=Path(config['TEST_DIR']).resolve();mode=config.get('MODE','submit')
    ensure(mode in {'submit','smoke'},'MODE must be submit or smoke.')
    allowed=Path('/kaggle/input') if mode=='submit' else Path('/kaggle/working')
    ensure(root.is_dir() and root.is_relative_to(allowed.resolve()),'Test directory outside the allowed Kaggle root for this mode.')
    visible=sorted(p for p in root.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(root).parts))
    unsupported=[p for p in visible if p.suffix.lower() in {'.dcm','.dicom','.gif','.jp2','.pgm','.pnm'}]
    files=[p for p in visible if p.suffix.lower() in IMAGE_EXTENSIONS]
    ensure(files or unsupported,'No image files found in TEST_DIR.')
    errors=[dict(Image_id=p.relative_to(root).as_posix(),path=str(p),error='Unsupported image format; obtain a BTC-approved reader/policy.') for p in unsupported]
    if not files:
        pd.DataFrame(errors,columns=['Image_id','path','error']).to_csv(out/'input_errors.csv',index=False)
        raise ValueError('Only unsupported images found; see input_errors.csv.')
    def normalized(value):return str(value).replace('\\','/')
    def identifier_stem(value):
        p=Path(value)
        return str(p.with_suffix('')) if p.suffix.lower() in IMAGE_EXTENSIONS else str(p)
    by_relative={p.relative_to(root).as_posix():p for p in files}
    indexes={kind:{} for kind in ['name','relative_case','name_case','relative_stem','stem_case']}
    for p in files:
        rel=p.relative_to(root).as_posix()
        for kind,key in [('name',p.name),('relative_case',rel.casefold()),('name_case',p.name.casefold()),('relative_stem',str(Path(rel).with_suffix('')).casefold()),('stem_case',p.stem.casefold())]:
            indexes[kind].setdefault(key,[]).append(p)
    mapping=[];paths=[];ids=[]
    if config.get('SAMPLE_SUBMISSION_CSV'):
        sample=pd.read_csv(config['SAMPLE_SUBMISSION_CSV'],usecols=['Image_id'],dtype=str,keep_default_na=False)
        ids=sample.Image_id.tolist()
        ensure(ids and len(set(ids))==len(ids) and all(x.strip() for x in ids),'Sample IDs must be unique and nonempty.')
        for original in ids:
            name=normalized(original);selected=None;method=None
            for kind,key in [('exact_relative',name),('name',name),('relative_case',name.casefold()),('name_case',name.casefold()),('relative_stem',identifier_stem(name).casefold()),('stem_case',Path(identifier_stem(name)).name.casefold())]:
                options=[by_relative[key]] if kind=='exact_relative' and key in by_relative else [] if kind=='exact_relative' else indexes[kind].get(key,[])
                if options:
                    ensure(len(options)==1,('Ambiguous sample Image_id; no prediction guessed',original,kind))
                    selected=options[0];method=kind;break
            ensure(selected is not None,('Sample ID has no matching image; no prediction guessed',original))
            paths.append(selected);mapping.append(dict(Image_id=original,relative_path=selected.relative_to(root).as_posix(),match_method=method))
        ensure(len(set(paths))==len(paths),'Multiple sample IDs resolve to the same physical image.')
    else:
        ensure(all(len(v)==1 for v in indexes['name'].values()),'Duplicate basenames require an authoritative BTC sample CSV.')
        paths=files;ids=[p.name for p in paths]
        mapping=[dict(Image_id=i,relative_path=p.relative_to(root).as_posix(),match_method='filename_without_sample') for i,p in zip(ids,paths)]
    pd.DataFrame(mapping).to_csv(out/'test_id_mapping.csv',index=False)
    extras=[p.relative_to(root).as_posix() for p in files if p not in set(paths)]
    save_json(dict(sample_is_authoritative=bool(config.get('SAMPLE_SUBMISSION_CSV')),extra_image_paths=extras,unsupported_image_paths=[str(p.relative_to(root)) for p in unsupported]),out/'input_scan_audit.json')
    if extras:print('WARNING sample IDs define output; extra images logged:',len(extras),flush=True)
    old_file=set(payload['seen_hashes']['file_sha256']);old_pixel=set(payload['seen_hashes']['pixel_sha256']);rows=[]
    for image_id,p in zip(ids,paths):
        try:
            image=read_xray(p);a=file_hash(p);b=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            rows.append(dict(Image_id=image_id,path=str(p),sha256=a,pixel_sha256=b,overlap_file=a in old_file,overlap_pixel=b in old_pixel))
        except Exception as error:errors.append(dict(Image_id=image_id,path=str(p),error=type(error).__name__+': '+str(error)))
    pd.DataFrame(errors,columns=['Image_id','path','error']).to_csv(out/'input_errors.csv',index=False)
    ensure(not errors,('Unreadable/unsupported images: no majority-class fallback, no complete submission generated; see input_errors.csv',len(errors)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'test_file_audit.csv',index=False)
    overlap=int((frame.overlap_file|frame.overlap_pixel).sum())
    if overlap:print('WARNING',overlap,'images overlap prior Train/Validation; predictions retained for every requested ID; BTC review required.',flush=True)
    return frame

def run_submission(config):
    require_kaggle();out=Path(config['OUTPUT']).resolve()
    ensure(out.is_relative_to(Path('/kaggle/working').resolve()),'OUTPUT must be under /kaggle/working.')
    out.mkdir(parents=True,exist_ok=True);mode=config.get('MODE','submit')
    team=config.get('TEAM_ID','team01');ensure(team and all(c.isalnum() or c in '_-' for c in team),'Invalid team ID.')
    final_name=team+'_submission.zip' if mode=='submit' else 'SMOKE_ONLY_'+team+'.zip'
    # Archive previous generated outputs so a failed rerun cannot expose a stale final CSV/ZIP.
    old=[out/name for name in [team+'.csv',final_name,'submission.ipynb','submission_metadata.json'] if (out/name).is_file()]
    if old:
        backup=out/'previous_runs'/str(time.time_ns());backup.mkdir(parents=True)
        for p in old:shutil.move(str(p),str(backup/p.name))
    save_json(dict(status='RUNNING',mode=mode),out/'run_status.json')
    try:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        source=resolve_model(config);net,payload=load_pickle(source,device)
        source_sha=file_hash(source);save_json(execution_environment(),out/'environment.json')
        frame=collect_test(config,payload,out)
        probabilities,_=predict(net,loader_for(frame,config),device,True)
        ensure(probabilities.image_id.tolist()==frame.Image_id.tolist(),'Prediction order differs from requested IDs.')
        labels=pd.DataFrame({'Image_id':probabilities.image_id,'Prediction':[LABEL_NAMES[int(x>=payload['threshold'])] for x in probabilities.p_pneumonia]})
        csv_path=out/(team+'.csv');validate_csv(labels,csv_path)
        probabilities.rename(columns={'image_id':'Image_id'}).to_csv(out/'test_probabilities_internal.csv',index=False)
        save_portable_pickle(payload,out/'model.pkl',payload.get('converted_from_sha256',source_sha))
        notebook=make_submission_notebook(config['_MODULE_SOURCE'])
        replay={k:v for k,v in config.items() if not k.startswith('_')};replay['MODEL_PATH']=''
        notebook['cells'][1]['source']='CONFIG='+repr(replay)+'\n'
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),'submission_cell_'+str(i),'exec')
        save_json(notebook,out/'submission.ipynb')
        overlap=int((frame.overlap_file|frame.overlap_pixel).sum())
        metadata=dict(version=VERSION,mode=mode,purpose='official_test_inference' if mode=='submit' else 'smoke_fixture_not_BTC_Test',
            seed=payload['seed'],weights=payload['weights'],threshold=payload['threshold'],n_images=len(frame),
            output_labels=['Normal','Pneumonia'],test_labels_used=False,overlap_images=overlap,
            eligibility_review_required=bool(overlap) or mode!='submit',independent_test_score_verified=False,
            model_source_sha256=source_sha,model_sha256=file_hash(out/'model.pkl'),csv_sha256=file_hash(csv_path),environment=execution_environment())
        save_json(metadata,out/'submission_metadata.json')
        dest=out/final_name;temporary=Path(str(dest)+'.tmp')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ['model.pkl',team+'.csv','submission.ipynb','submission_metadata.json','test_file_audit.csv','test_id_mapping.csv','input_scan_audit.json','environment.json']:z.write(out/name,name)
        with zipfile.ZipFile(temporary) as z:ensure(z.testzip() is None,'ZIP CRC failed.')
        os.replace(temporary,dest)
        status='SMOKE_ONLY' if mode=='smoke' else 'GENERATED_REVIEW_REQUIRED' if overlap else 'GENERATED'
        save_json(dict(status=status,zip=str(dest),n_images=len(frame),eligibility_review_required=metadata['eligibility_review_required']),out/'run_status.json')
        print(status,str(dest),'rows',len(frame),'threshold0.55 | no Test score claimed',flush=True)
        return dest
    except Exception as error:
        save_json(dict(status='FAILED',mode=mode,error=type(error).__name__+': '+str(error),previous_outputs_archived=True),out/'run_status.json')
        raise

def verify_export(config,module_source):
    require_kaggle(t4=True);device=torch.device('cuda:0');out=Path(config['OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    seed=2026;selection,threshold,selection_source=find_selection(seed)
    assert threshold==.55,'The candidate threshold must remain locked to the previous analysis.'
    run_dirs=[find_run(a,seed) for a in ['resnet18','resnet18_eca']]
    manifests=[(d/'split_manifest.csv').read_bytes() for d in run_dirs]
    assert manifests[0]==manifests[1]
    manifest=pd.read_csv(io.BytesIO(manifests[0]),keep_default_na=False)
    fingerprint=manifest_fingerprint(manifest)
    assert fingerprint=='9f9474fafb054f501535269fa3e55bea4c3e24d794f6afe9c019abe6ce798ac2'
    val=manifest[manifest.split=='validation'].sort_values('image_id').reset_index(drop=True)
    assert len(val)==1028 and val.image_id.is_unique and len(manifest)==5132
    assert not set(manifest.loc[manifest.split=='train','pixel_sha256'])&set(val.pixel_sha256)
    saved=[];references=[];provenance=[];nets=[];source_metrics=[]
    for arch,d in zip(['resnet18','resnet18_eca'],run_dirs):
        p=d/'best.pt';item=validate_member(torch.load(p,map_location='cpu',weights_only=True),arch,seed,fingerprint)
        saved.append(item);net=network_from_state(item,item['state_dict']).to(device).eval();nets.append(net)
        reference=pd.read_csv(d/'val_predictions.csv').sort_values('image_id').reset_index(drop=True)
        assert reference.image_id.tolist()==val.image_id.tolist() and np.array_equal(reference.true_label.to_numpy(),val.label.to_numpy())
        references.append(reference);source_metrics.append(json.loads((d/'metrics.json').read_text()))
        provenance.append(dict(architecture=arch,seed=seed,checkpoint=str(p),checkpoint_sha256=file_hash(p),
            predictions_sha256=file_hash(d/'val_predictions.csv'),manifest_sha256=file_hash(d/'split_manifest.csv'),best_epoch=item['best_epoch']))
    assert saved[0]['initial_backbone_hash']==saved[1]['initial_backbone_hash']
    root=resolve_data_root(config,manifest)
    print('CHECKPOINT_PAIR_OK | seed2026 | baseline + ECA | weights 0.5/0.5 | threshold',threshold,flush=True)
    # Verify the actual BTC image bytes and normalized pixels against the original manifest.
    for count,r in enumerate(val.itertuples(index=False),1):
        rel=Path(str(r.relative_path));assert not rel.is_absolute() and '..' not in rel.parts
        p=root/rel;assert file_hash(p)==r.sha256,'Validation image bytes changed.'
        image=read_xray(p);assert hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()==r.pixel_sha256
        if count%250==0:print('IMAGE_AUDIT',count,'/ 1028',flush=True)
    print('IMAGE_AUDIT_OK | 1028/1028 | original preprocessing',flush=True)
    environment=dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),python=platform.python_version(),
                     device=str(device),gpu=torch.cuda.get_device_name(0),visible_gpu_count=torch.cuda.device_count(),
                     used_gpu_indices=[0],precision='float32',new_training=False,pretrained=False)
    save_json(environment,out/'environment.json')
    loader=loader_for(val,config,root);actual=[];checks={};metric_rows=[]
    for arch,net,reference,sm in zip(['resnet18','resnet18_eca'],nets,references,source_metrics):
        frame,_=predict(net,loader,device)
        checks[arch]=compare_probabilities(frame,reference,float(sm['validation_tuned']['threshold']))
        actual.append(frame)
        m=metrics(val.label,frame.p_pneumonia,float(sm['validation_tuned']['threshold']))
        for k,v in sm['validation_tuned'].items():
            if k in m:assert np.isclose(m[k],v,atol=1e-12,rtol=0)
        metric_rows.append(dict(model=arch,**m))
        print('ORIGINAL_PROBABILITIES_MATCH',arch,json.dumps(checks[arch]),flush=True)
    model_path=out/'model.pkl'
    payload=export_pickle(saved,[.5,.5],threshold,manifest,model_path,provenance)
    loaded,payload=load_pickle(model_path,device)
    reloaded,_=predict(loaded,loader,device,True)
    expected=references[0][['image_id','p_pneumonia']].copy()
    expected['p_pneumonia']=(references[0].p_pneumonia.to_numpy()+references[1].p_pneumonia.to_numpy())/2
    checks['ensemble_pickle_reload']=compare_probabilities(reloaded,expected,threshold)
    previous_tables=list(Path('/kaggle/input').rglob('metrics_validation_tuned.csv'))
    assert len(previous_tables)==1,'Attach the Analysis Output for the exact existing candidate comparison.'
    previous=pd.read_csv(previous_tables[0]);r=previous[(previous.method==selection['method'])&(previous.seed==seed)]
    assert len(r)==1
    ensemble_metrics=metrics(val.label,reloaded.p_pneumonia,threshold)
    for k,v in ensemble_metrics.items():assert np.isclose(v,float(r.iloc[0][k]),atol=1e-12,rtol=0),k
    metric_rows.append(dict(model='ensemble_50_50',**ensemble_metrics))
    print('MODEL_PKL_FULL_VALIDATION_RELOAD_OK',json.dumps(checks['ensemble_pickle_reload']),flush=True)
    detailed=val[['image_id','label']].copy()
    for name,frame in zip(['baseline_probability','eca_probability','ensemble_probability'],actual+[reloaded]):detailed[name]=frame.p_pneumonia.to_numpy()
    detailed['ensemble_prediction']=[LABEL_NAMES[int(x>=threshold)] for x in reloaded.p_pneumonia]
    detailed.to_csv(out/'validation_predictions_reloaded.csv',index=False,float_format='%.17g')
    schema=pd.DataFrame({'Image_id':reloaded.image_id,'Prediction':detailed.ensemble_prediction})
    validate_csv(schema,out/'validation_format_check.csv')
    pd.DataFrame(metric_rows).to_csv(out/'validation_metrics.csv',index=False)
    batch,_=next(iter(loader));forward=benchmark_forward({'resnet18':loaded.members[0],'ensemble_50_50':loaded},batch,device)
    e2e=[]
    # Both have already processed the entire set. Alternate order across two measured passes.
    for repeat,order in enumerate([['resnet18','ensemble_50_50'],['ensemble_50_50','resnet18']],1):
        for name in order:
            net=loaded.members[0] if name=='resnet18' else loaded
            frame,seconds=predict(net,loader,device,name!='resnet18')
            compare_probabilities(frame,references[0] if name=='resnet18' else expected,
                                  float(source_metrics[0]['validation_tuned']['threshold']) if name=='resnet18' else threshold)
            e2e.append(dict(model=name,repeat=repeat,n_images=len(frame),seconds=seconds,ms_per_image=1000*seconds/len(frame),
                            images_per_second=len(frame)/seconds,scope='disk read + exif/grayscale/pad/normalize + DataLoader + H2D + forward + D2H; warmed cache, two alternating-order passes'))
    size=dict(model_pkl_bytes=model_path.stat().st_size,model_pkl_MiB=model_path.stat().st_size/1024**2,
              baseline_parameters=sum(p.numel() for p in loaded.members[0].parameters()),
              eca_parameters=sum(p.numel() for p in loaded.members[1].parameters()),
              ensemble_parameters=sum(p.numel() for p in loaded.parameters()),
              baseline_state_tensor_bytes=sum(v.numel()*v.element_size() for v in saved[0]['state_dict'].values()),
              ensemble_state_tensor_bytes=sum(v.numel()*v.element_size() for s in saved for v in s['state_dict'].values()))
    assert size['ensemble_parameters']==22354055
    bench=dict(environment=environment,forward=forward,end_to_end=e2e,size=size)
    save_json(bench,out/'benchmark.json');pd.DataFrame(forward).to_csv(out/'benchmark_forward.csv',index=False)
    pd.DataFrame(e2e).to_csv(out/'benchmark_end_to_end.csv',index=False)
    save_json(dict(status='passed',candidate_selection_source=selection_source,seed=seed,weights=[.5,.5],threshold=threshold,
        n_validation=len(val),image_bytes_and_pixels_match_manifest=True,checks=checks,
        model_sha256=file_hash(model_path),no_new_training=True,test_labels_used=False,independent_test_verified=False),out/'verification.json')
    save_json(provenance,out/'source_checkpoints.json')
    (out/'ensemble_inference.py').write_text(module_source,encoding='utf-8')
    submission=make_submission_notebook(module_source)
    save_json(submission,out/'submission.ipynb')
    save_json(make_verification_notebook(config,module_source),out/'verify_export.ipynb')
    report=['# Ensemble ResNet18 gốc + ECA seed 2026 đã kiểm tra trọng số','',
      'Cấu hình: ghép xác suất 50/50, ngưỡng 0,55 đã khóa từ phân tích validation trước. Dùng hai checkpoint cùng seed 2026, không train lại. Phiên kiểm tra dùng GPU Tesla T4, FP32, GPU số 0.',
      '', '## Kiểm tra đã đạt','',
      '- Hai checkpoint đúng kiến trúc/seed, cùng manifest và backbone khởi tạo.',
      '- Cả 1.028 ảnh validation khớp byte và pixel với manifest cũ.',
      '- Xác suất từng model khớp CSV nguồn (atol 1e-6, rtol 1e-5), nhãn không thay đổi.',
      '- model.pkl chứa cả hai bộ trọng số dạng NumPy và metadata. Đã nạp lại model.pkl và suy luận toàn bộ validation, khớp ensemble CSV và confusion matrix đã đo.',
      '- CSV kiểm tra định dạng có đúng Image_id, Prediction. Đây là validation_format_check.csv, không phải bài nộp Test.',
      '', '## Chỉ số validation của một cặp seed 2026','',
      '| Model | Ngưỡng | P (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) | FN | FP |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in metric_rows:report.append(f"| {r['model']} | {r['threshold']:.2f} | {100*r['P']:.4f} | {100*r['Accuracy']:.4f} | {100*r['Sensitivity']:.4f} | {100*r['Specificity']:.4f} | {r['FN']} | {r['FP']} |")
    report+=['','## Tốc độ và dung lượng','','| Model | Batch | Forward median (ms/batch) | P95 (ms/batch) | Images/s |','|---|---:|---:|---:|---:|']
    for r in forward:report.append(f"| {r['model']} | {r['batch_size']} | {r['median_batch_ms']:.3f} | {r['p95_batch_ms']:.3f} | {r['throughput_images_per_second']:.1f} |")
    for name in ['resnet18','ensemble_50_50']:
        times=[r['seconds'] for r in e2e if r['model']==name]
        report.append(f"\n{name}: median end-to-end trên 1.028 ảnh, hai lượt đo thứ tự luân phiên = {np.median(times):.3f} giây ({1000*np.median(times)/1028:.3f} ms/ảnh theo thông lượng batch32).")
    report += ['',f"model.pkl: {size['model_pkl_bytes']:,} byte = {size['model_pkl_MiB']:.2f} MiB. Tổng {size['ensemble_parameters']:,} tham số. Forward-only không gồm đọc ảnh/tiền xử lý. End-to-end có đọc ảnh/tiền xử lý, warm cache, batch32, workers2; ms/ảnh ở đây là thời gian trung bình theo batch, không phải độ trễ ảnh đơn. Batch1 mới phản ánh forward ảnh đơn.",
      '', '## Cách dùng và lưu file','',
      'Giải nén vào PneumoScore-XAI/Outputs/Ensemble_Seed2026_Verified/. Giữ toàn bộ thư mục và model/resume cũ. Có thể dọn bản trùng trong Downloads sau khi kiểm tra bản đã lưu.',
      '', '- model.pkl: trọng số hai model và cấu hình ghép/ngưỡng, cần giữ.',
      '- submission.ipynb: mở trên Kaggle, Add Input model.pkl và Test BTC, điền TEST_DIR/mã đội/CSV mẫu, Run All để tự sinh CSV và ZIP bài nộp. Không sửa CSV bằng tay.',
      '- verify_export.ipynb: tái lập phiên kiểm tra trên Kaggle T4; cần Output Stability, Output Ensemble Analysis và dataset BTC đã dùng.',
      '- ensemble_inference.py: mã trong notebook để tham khảo và giải thích, không cần chạy riêng trên Windows.',
      '- validation_*.csv, validation_metrics.csv, verification.json: bằng chứng suy luận lại và kiểm tra định dạng, không nộp làm kết quả Test.',
      '- benchmark*.csv/json, environment.json, source_checkpoints.json và báo cáo này: lưu hồ sơ nghiên cứu.',
      '', 'Gói hiện tại là bộ model đã kiểm tra, chưa phải ZIP bài nộp vì chưa có ảnh Test BTC trong phiên này. Khi có Test, notebook sinh một .pkl, một .ipynb và team01.csv theo Điều 6. Không dùng nhãn Test.',
      '', 'P của cặp seed 2026 không phải trung bình ba seed. Các chỉ số vẫn là validation đã dùng chọn checkpoint/ensemble/ngưỡng; không chứng minh hiệu quả trên bệnh nhân mới. Giữ baseline làm phương án chính đã chuẩn hóa và ensemble làm ứng viên đối chứng.']
    (out/'README_VI.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    dest=out.parent/'PneumoScore_Ensemble_Seed2026_Verified.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,'Ensemble_Seed2026_Verified/'+p.name)
    print('VALIDATION_METRICS\n'+pd.DataFrame(metric_rows).to_string(index=False),flush=True)
    print('BENCHMARK_FORWARD\n'+pd.DataFrame(forward).to_string(index=False),flush=True)
    print('BENCHMARK_E2E\n'+pd.DataFrame(e2e).to_string(index=False),flush=True)
    print('MODEL_SIZE',json.dumps(size),flush=True)
    print('ENSEMBLE_EXPORT_VERIFIED_DONE',str(dest),flush=True)
    return dest
