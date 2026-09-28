"""
Reactive Strength Index (for Continuous CMJ) Automated Analysis System
Low-cost, zero-hardware quantitative assessment based on monocular vision
"""
import streamlit as st, cv2, numpy as np, tempfile, os, time, pandas as pd
from core.factory import create_detector, create_pose, create_rsi_calculator
from core.config import DetectorConfig, PoseConfig, RSIConfig
from core.rsi.helpers import hip_center_y, ankle_center_y, estimate_trochanter_height
from core.device import get_device_info

st.set_page_config(page_title="🦘 Reactive Strength Index (for Continuous CMJ) Automated Analysis System", layout="wide", page_icon="🦘")

# ---- Sidebar ----
with st.sidebar:
    # Auto GPU detection display
    dev_info = get_device_info()
    st.caption(f"Device: **{dev_info['label']}** (`{dev_info['device']}`) — auto-detected for Apple M-chip")
    st.header("Parameters")
    conf = st.slider("Detection Confidence", 0.1, 0.6, 0.30, 0.05, help="Lower to 0.25 for small targets / dim light")
    iou = st.slider("NMS IoU", 0.3, 0.7, 0.45, 0.05)
    enable_downsample = st.checkbox("Enable Downsampling", value=False, help="Default: max native fps; when enabled, selectable 60fps-native")
    if enable_downsample:
        st.caption("Downsampling range: min 60fps, max native (selectable after upload)")
    else:
        st.caption("FPS: max native (no downsampling) recommended for contact precision")
    lowcut = st.slider("Bandpass Lowcut Hz", 0.1, 1.0, 0.3, 0.1)
    highcut = st.slider("Bandpass Highcut Hz", 4.0, 12.0, 8.0, 0.5)
    prom = st.slider("Peak Prominence", 0.2, 0.8, 0.5, 0.05)
    dist = st.slider("Min Jump Interval s", 0.2, 0.8, 0.35, 0.05, help="Continuous CMJ ~0.4-0.8s/jump")
    show_bbox = st.checkbox("Show Bbox", True)
    show_skeleton = st.checkbox("Show Skeleton", True)
    st.divider()
    st.subheader("Jump Height Correction")
    user_h = st.number_input("Height cm", 120, 220, 180, 1, help="Used with trochanter for prediction")
    trochanter = st.number_input("Trochanter Height cm (manual)", 60.0, 120.0, float(estimate_trochanter_height(180)), 0.5, help="Measure trochanter to ground with tape")
    st.caption(f"Input: height {user_h}cm, trochanter {trochanter:.1f}cm (for reference only)")
    st.warning("Jump height is physics-based on flight time + trochanter pixel scaling, **prediction only, not accurate**", icon="⚠️")
    st.divider()
    st.markdown("**Two-stage**\n- Stage1 YOLOv8n person bbox\n- Stage2 YOLOv8n-Pose hip/ankle y\n- Contact: ankle collapse + shoe deformation proxy\n- FPS: full native maximized")

@st.cache_resource
def get_models(c,i): return create_detector(DetectorConfig(conf=c,iou=i)), create_pose(PoseConfig())
detector, pose = get_models(conf, iou)
rsi_calc = create_rsi_calculator(RSIConfig(lowcut=lowcut, highcut=highcut, prominence=prom, distance_sec=dist))

st.title("🦘 Reactive Strength Index (for Continuous CMJ) Automated Analysis System")
st.caption("Low-cost, zero-hardware solution based on monocular vision · Alternative to Optojump/force plates, enabling automated quantitative analysis of continuous CMJ per-jump RSI, contact/flight time and height | Proof-of-Concept")

upl, info = st.columns([2,1])
with upl:
    uploaded = st.file_uploader("Upload Video (recommended side view, 5-10 CMJs)", type=["mp4","avi","mov","mkv","webm"])
with info:
    st.info("Shooting: tripod side view 1.5-3m recommended (higher accuracy, front view also works), single person centered 15%+, 5-10 continuous jumps")
    st.markdown("Tech: `YOLOv8n + YOLOv8n-Pose + hip vertical trajectory`")

if uploaded is None:
    st.warning("Please upload video. Synthetic demo preview:")
    with st.expander("Synthetic Demo 5 jumps 60fps", expanded=True):
        fs=60; time_axis=np.linspace(0,5,300)
        hip=[0.6-0.12*np.sin(2*np.pi*1.25*x) for x in time_axis]
        ankle=[0.85+0.08*np.sin(2*np.pi*1.25*x+np.pi) for x in time_axis]
        res=rsi_calc.calculate(hip, fs, ankle_signal=ankle, user_height_cm=user_h, trochanter_cm=trochanter)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Jump Count",f"{res['jump_count']}"); c2.metric("Mean RSI",f"{res['mean_rsi']:.2f}"); c3.metric("Mean RSImod",f"{res['mean_rsi_mod']:.2f}"); c4.metric("Best RSI",f"{res['best_rsi']:.2f}"); c5.metric("Contact/Flight/TT T",f"{res['mean_contact']:.3f}/{res['mean_flight']:.3f}/{res['mean_ttt']:.3f}s")
        if res['jump_count']:
            df_show=pd.DataFrame(res['jumps'])[["id","t_flight","t_contact","t_ttt","rsi","rsi_mod","rsi_mod_contact","height"]].copy()
            df_show["height"]=df_show["height"].round(3)
            df_show["rsi"]=df_show["rsi"].round(3); df_show["rsi_mod"]=df_show["rsi_mod"].round(3)
            st.dataframe(df_show, use_container_width=True)
            st.line_chart(pd.DataFrame({"filtered":res['filtered']}))
            st.line_chart(pd.DataFrame({"RSI":[j['rsi'] for j in res['jumps']],"RSImod":[j['rsi_mod'] for j in res['jumps']]}, index=[j['id'] for j in res['jumps']]))

if uploaded is not None:
    tmp=tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]); tmp.write(uploaded.read()); tmp.flush()
    cap=cv2.VideoCapture(tmp.name)
    if not cap.isOpened(): st.error("Failed to read video"); st.stop()
    orig_fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; cnt=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.success(f"{uploaded.name} | {W}x{H} | {orig_fps:.0f}fps | {cnt} frames")
    if enable_downsample and orig_fps>60:
        target=st.slider("Processing FPS 60-native",60,int(orig_fps),int(orig_fps),5)
        stride=max(1,int(round(orig_fps/target))); actual=float(orig_fps/stride)
        st.caption(f"Downsampled: {orig_fps:.0f}→{actual:.0f}fps every {stride} frame")
    else:
        if enable_downsample and orig_fps<=60: st.info(f"Native ≤60fps, already maximized")
        stride=1; actual=float(orig_fps); st.caption(f"Maximized: {actual:.0f}fps full-frame")
    if st.button("Start Analysis", type="primary", use_container_width=True):
        detector.reset(); prog=st.progress(0); prev=st.empty()
        out_path=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4").name
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'); out=cv2.VideoWriter(out_path, fourcc, actual, (W,H))
        if not out.isOpened():
            fourcc=cv2.VideoWriter_fourcc(*'avc1'); out=cv2.VideoWriter(out_path, fourcc, actual, (W,H))
        hip_sig, ankle_sig = [],[]; idx=0; proc=valid=0; t0=time.time()
        preview_n=max(1,cnt//20)
        while True:
            ok,frame=cap.read()
            if not ok: break
            if idx%stride!=0: idx+=1; continue
            bbox=detector.detect(frame); ann=frame.copy()
            h_y=a_y=None
            if bbox:
                if show_bbox: cv2.rectangle(ann,(bbox.x1,bbox.y1),(bbox.x2,bbox.y2),(0,255,0),2)
                roi=frame[max(0,bbox.y1):min(H,bbox.y2), max(0,bbox.x1):min(W,bbox.x2)]
                if roi.size and roi.shape[0]>0:
                    if roi.shape[0]<256:
                        s=256/roi.shape[0]; roi=cv2.resize(roi,(int(roi.shape[1]*s),256))
                    kps=pose.estimate(roi)
                    if kps:
                        h_y=hip_center_y(kps); a_y=ankle_center_y(kps); valid+=1
                        if show_skeleton: ann=pose.draw(ann,kps,bbox)
            hip_sig.append(h_y); ankle_sig.append(a_y); out.write(ann); proc+=1
            if proc%10==0 or idx%preview_n==0:
                prog.progress(min(0.95,proc/max(1,cnt//stride)), text=f"Processing {proc}/{cnt//stride} valid {valid}")
                if idx%preview_n==0: prev.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB), caption=f"frame {idx}", width=420)
            idx+=1
        cap.release(); out.release()
        prog.progress(1.0, text=f"Done {proc} frames {time.time()-t0:.1f}s valid {valid}"); st.metric("Valid Pose Rate",f"{valid/max(1,proc)*100:.1f}%", help=">70% good, <30% re-shoot recommended")
        res=rsi_calc.calculate(hip_sig, fs=actual, ankle_signal=ankle_sig, user_height_cm=user_h, trochanter_cm=trochanter)
        if res["jump_count"]==0:
            st.error(f"No complete jump detected: {res['message']}"); st.line_chart(pd.DataFrame({"hip_y":[0 if v is None else v for v in hip_sig]})); st.stop()
        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("Jump Count",f"{res['jump_count']}"); m2.metric("Mean RSI",f"{res['mean_rsi']:.2f}"); m3.metric("Mean RSImod",f"{res['mean_rsi_mod']:.2f}"); m4.metric("Best RSI",f"{res['best_rsi']:.2f}"); m5.metric("Contact/Flight/TT T",f"{res['mean_contact']:.3f}/{res['mean_flight']:.3f}/{res['mean_ttt']:.3f}s"); m6.metric("Height (m)",f"{res['mean_height']:.3f} m")
        st.caption(f"Contact: {res['contact_method']} (ankle collapse+shoe deform) | Trochanter {res['trochanter_cm']:.1f}cm manual | ⚠️ Prediction only")
        df=pd.DataFrame(res['jumps']); df_show=df[["id","t_contact","t_flight","t_ttt","rsi","rsi_mod","rsi_mod_contact","height"]].copy()
        df_show["height"]=df_show["height"].round(3)
        st.dataframe(df_show, use_container_width=True)
        t1,t2,t3=st.tabs(["RSI Curves","Waveform","Annotated Video"])
        with t1:
            st.line_chart(pd.DataFrame({"RSI":[j['rsi'] for j in res['jumps']],"RSImod":[j['rsi_mod'] for j in res['jumps']]}, index=[j['id'] for j in res['jumps']]))
            st.line_chart(pd.DataFrame({"Height (m)":[j['height'] for j in res['jumps']]}, index=[j['id'] for j in res['jumps']]))
        with t2: st.line_chart(pd.DataFrame({"filtered":res['filtered']})); st.caption(f"peaks {len(res['peaks'])} troughs {len(res['troughs'])}")
        with t3:
            if os.path.exists(out_path) and os.path.getsize(out_path)>0:
                st.video(out_path)
                with open(out_path,"rb") as f: st.download_button("Download Annotated Video",f,file_name="rsi_annotated.mp4",mime="video/mp4",use_container_width=True)
        c1,c2=st.columns(2)
        c1.download_button("Download Frame CSV", pd.DataFrame({"frame":range(len(hip_sig)),"hip_y":[0 if v is None else v for v in hip_sig],"filtered":list(res['filtered'])}).to_csv(index=False).encode(), file_name="frames.csv", use_container_width=True)
        c2.download_button("Download Jump CSV", pd.DataFrame(res['jumps']).to_csv(index=False).encode(), file_name="jumps.csv", use_container_width=True)
        try: os.unlink(tmp.name)
        except: pass

st.divider()
st.caption("Proof-of-Concept: YOLOv8n (det) + YOLOv8n-Pose (pose) off-the-shelf small models · Low-cost zero-hardware · Validated on 5-jump synthetic and small-target real footage")
