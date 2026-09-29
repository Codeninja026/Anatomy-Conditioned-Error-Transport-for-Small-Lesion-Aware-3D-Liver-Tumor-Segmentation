"""Extract 3DIRCADb-01 ZIP DICOM archives into patient-level NIfTI files.

This is an evaluation preparation step only. IRCADb must remain completely
unseen during model selection and threshold tuning.
"""
from __future__ import annotations
import argparse, io, zipfile
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom

def read_series(z, prefix):
    files=[n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")]
    rows=[]
    for name in files:
        ds=pydicom.dcmread(io.BytesIO(z.read(name)), force=True)
        order=float(getattr(ds,"InstanceNumber",len(rows)))
        if hasattr(ds,"ImagePositionPatient"): order=float(ds.ImagePositionPatient[2])
        rows.append((order,ds))
    rows.sort(key=lambda x:x[0]); return rows

def convert_patient(patient_dir: Path, out_dir: Path):
    patient=patient_dir.name; out_dir.mkdir(parents=True,exist_ok=True)
    image_zip=zipfile.ZipFile(patient_dir/"PATIENT_DICOM.zip")
    images=read_series(image_zip,"PATIENT_DICOM/")
    if not images: raise RuntimeError(f"No patient DICOMs in {patient}")
    volume=np.stack([d.pixel_array for _,d in images],axis=-1).astype(np.float32)
    masks=np.zeros(volume.shape,dtype=np.uint8)
    mask_zip=zipfile.ZipFile(patient_dir/"MASKS_DICOM.zip")
    liver=read_series(mask_zip,"MASKS_DICOM/liver/")
    if liver:
        liver_arr=np.stack([d.pixel_array for _,d in liver],axis=-1)
        masks[:,:,:min(masks.shape[2],liver_arr.shape[2])][liver_arr[:,:,:min(masks.shape[2],liver_arr.shape[2])]>0]=1
    for n in sorted(set(x.split('/')[1] for x in mask_zip.namelist() if x.startswith("MASKS_DICOM/livertumor"))):
        series=read_series(mask_zip,f"MASKS_DICOM/{n}/")
        if series:
            arr=np.stack([d.pixel_array for _,d in series],axis=-1)
            depth=min(masks.shape[2],arr.shape[2]); masks[:,:,:depth][arr[:,:,:depth]>0]=2
    d=images[0][1]; ps=getattr(d,"PixelSpacing",[1.0,1.0]); sz=float(getattr(d,"SliceThickness",1.0))
    affine=np.diag([float(ps[0]),float(ps[1]),sz,1.0])
    nib.save(nib.Nifti1Image(volume,affine),out_dir/f"volume-{patient}.nii.gz")
    nib.save(nib.Nifti1Image(masks,affine),out_dir/f"segmentation-{patient}.nii.gz")
    return volume.shape, int((masks==2).sum())

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="dataset/3Dircadb1"); p.add_argument("--output",default="prepared/3Dircadb1"); a=p.parse_args()
    root=Path(a.root); out=Path(a.output); rows=[]
    for d in sorted(root.glob("3Dircadb1.*")):
        shape,tumor=convert_patient(d,out/d.name); rows.append({"patient":d.name,"shape":shape,"tumor_voxels":tumor}); print(d.name,shape,tumor)
    (out/"manifest.json").write_text(__import__('json').dumps(rows,indent=2),encoding="utf-8")

if __name__=="__main__": main()
