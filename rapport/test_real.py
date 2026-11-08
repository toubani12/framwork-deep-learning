import time, json, numpy as np, tensorflow as tf
from PIL import Image, ImageOps
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CLASSES=["daisy","dandelion","roses","sunflowers","tulips"]
t=time.time(); m=tf.keras.models.load_model("modele_flowers.keras"); tload=time.time()-t
print("load",tload, m.name, m.count_params())
tr=sum(int(np.prod(w.shape)) for w in m.trainable_weights)
print("trainable",tr)
tests=[("types_images/daisy.jpeg","daisy"),("types_images/dandelion.jpeg","dandelion"),("types_images/rose.jpeg","roses"),
("types_images/sunflower.jpeg","sunflowers"),("types_images/tulipe.jpeg","tulips"),
("images.jpeg","?"),("images (1).jpeg","?"),("images (2).jpeg","?"),("test_dandelion_user.jpg","dandelion"),("Blood_orange_on_white.jpg","hors-classe (orange)")]
res=[]
for p,true in tests:
    img=ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    x=tf.image.resize(tf.keras.utils.img_to_array(img),[224,224])
    b=tf.stack([x,tf.image.flip_left_right(x)])
    t=time.time(); pr=m.predict(b,verbose=0).mean(0); dt=time.time()-t
    k=int(pr.argmax()); res.append(dict(file=p,true=true,pred=CLASSES[k],conf=float(pr[k]),ms=dt*1000,probs=[float(v) for v in pr]))
    print(p,true,CLASSES[k],f"{pr[k]:.3f}",f"{dt*1000:.0f}ms",np.round(pr,3))
json.dump(dict(load=tload,params=m.count_params(),trainable=tr,res=res),open("out/tests.json","w"),indent=1)
fig,axes=plt.subplots(2,len(res),figsize=(2.3*len(res),4.6),gridspec_kw={"height_ratios":[1.1,1]})
for i,r in enumerate(res):
    axes[0,i].imshow(Image.open(r["file"]).convert("RGB")); axes[0,i].axis("off")
    axes[0,i].set_title(f"{r['pred']}\n{r['conf']:.1%}",fontsize=9,color="#1E8449" if r['conf']>=0.6 else "#C0392B")
    axes[1,i].barh(CLASSES,r["probs"],color=["#1E8449" if c==r['pred'] else "#BDC3C7" for c in CLASSES]); axes[1,i].set_xlim(0,1); axes[1,i].invert_yaxis()
    axes[1,i].tick_params(labelsize=7)
    if i: axes[1,i].set_yticklabels([])
plt.tight_layout(); plt.savefig("out/tests_grid.png",dpi=150)
