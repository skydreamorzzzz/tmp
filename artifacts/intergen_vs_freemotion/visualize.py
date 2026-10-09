"""Fixed physical-coordinate skeleton GIFs; no smoothing or actor shifting."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

OUT=Path(__file__).resolve().parent
CHAINS=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]

def draw(ax,joints,lims):
    for p,c in enumerate(['#cb3445','#2279b4'][:len(joints)]):
        for chain in CHAINS:
            q=joints[p,chain]
            ax.plot(q[:,0],q[:,2],q[:,1],color=c,linewidth=2)
    ax.set_xlim(*lims[0]);ax.set_ylim(*lims[1]);ax.set_zlim(*lims[2])
    ax.set_xlabel('X (m)');ax.set_ylabel('Z (m)');ax.set_zlabel('Y (m)')
    ax.view_init(elev=16,azim=-65)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',default='native_two_person');args=p.parse_args()
    rows=json.loads((OUT/'test_manifest.json').read_text())['rows'];v=OUT/'visualizations';v.mkdir(exist_ok=True)
    chosen=rows[:3];observations=[]
    for row in chosen:
        path=OUT/args.mode/(row['id']+'.npy')
        if not path.exists():continue
        m=np.load(path);j=m[...,:66].reshape(len(m),2,22,3)
        # Single animations show actor A alone; paired single renders separately
        # expose any overlap when both independent single outputs are combined.
        if args.mode=='single':j=j[:,:1]
        low=j.min(axis=(0,1,2));high=j.max(axis=(0,1,2));center=(low+high)/2
        radius=max(high-low)/2+.15
        lims=[(center[k]-radius,center[k]+radius) for k in [0,2,1]]
        frames=[]
        for t in range(0,len(m),2):
            fig=plt.figure(figsize=(5,5),dpi=80);ax=fig.add_subplot(111,projection='3d');draw(ax,j[t],lims)
            label='InterGen zero-B single' if args.mode=='single' else 'InterGen native two-person'
            fig.suptitle(f"{label} | test {row['id']}\nframe {t}/{len(m)}",fontsize=10)
            fig.tight_layout();fig.canvas.draw();w,h=fig.canvas.get_width_height()
            frames.append(Image.frombytes('RGB',(w,h),fig.canvas.tostring_rgb()));plt.close(fig)
        dest=v/(args.mode+'_'+row['id']+'.gif')
        frames[0].save(dest,save_all=True,append_images=frames[1:],duration=67,loop=0)
        fig=plt.figure(figsize=(15,3.3),dpi=100)
        for i,t in enumerate(np.linspace(0,len(m)-1,5,dtype=int)):
            ax=fig.add_subplot(1,5,i+1,projection='3d');draw(ax,j[t],lims);ax.set_title('frame '+str(t),fontsize=9)
        caption=row['text1'] if args.mode=='single' else row['text']
        fig.suptitle(f"{label} | test {row['id']}\n{caption}",fontsize=10)
        fig.tight_layout(rect=[0,0,1,.86]);fig.savefig(v/(args.mode+'_'+row['id']+'.png'));plt.close(fig)
        if args.mode=='single':
            paired=m[...,:66].reshape(len(m),2,22,3)
            low=paired.min(axis=(0,1,2));high=paired.max(axis=(0,1,2));center=(low+high)/2
            r=max(high-low)/2+.15;pairlims=[(center[k]-r,center[k]+r) for k in [0,2,1]]
            fig=plt.figure(figsize=(15,3.3),dpi=100)
            for i,t in enumerate(np.linspace(0,len(m)-1,5,dtype=int)):
                ax=fig.add_subplot(1,5,i+1,projection='3d');draw(ax,paired[t],pairlims);ax.set_title('frame '+str(t),fontsize=9)
            fig.suptitle(f"Two independent InterGen zero-B singles paired | test {row['id']}\n{row['text']}",fontsize=10)
            fig.tight_layout(rect=[0,0,1,.86]);fig.savefig(v/('single_paired_'+row['id']+'.png'));plt.close(fig)
        bones=np.concatenate([np.linalg.norm(j[:,:,np.array(c[1:])]-j[:,:,np.array(c[:-1])],axis=-1) for c in CHAINS],axis=-1)
        observations.append({'id':row['id'],'text':row['text'],'length':len(m),'gif':str(dest.relative_to(OUT)),
                             'finite':bool(np.isfinite(m).all()),'bone_length_mean_m':float(bones.mean()),
                             'root_distance_mean_m':float(np.linalg.norm(m[:,0,:3]-m[:,1,:3],axis=-1).mean()),
                             'mean_joint_frame_displacement_m':float(np.linalg.norm(np.diff(j,axis=0),axis=-1).mean())})
    (v/(args.mode+'_cases.json')).write_text(json.dumps(observations,indent=2)+'\n')
    print(json.dumps(observations,indent=2))
if __name__=='__main__':main()
