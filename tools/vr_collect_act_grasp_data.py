#!/usr/bin/env python3
"""MuJoCo VR left-arm ACT grasp collector."""
import argparse,cv2,h5py,json,os,select,shutil,sys,tempfile,termios,threading,time,tty
from pathlib import Path
import numpy as np
import yaml
WS=Path(__file__).resolve().parents[1];U="/home/kiorobot/kio_robot_zzc/kio_upoo-main";D=Path("/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")
for p in(str(D),str(D/"television"),U):
 if p not in sys.path:sys.path.insert(0,p)
import mujoco
import mujoco.viewer
from TeleVision import OpenTeleVision
from constants_vuer import grd_yup2grd_zup
from motion_utils import mat_update,fast_mat_inv
from upoo_cartesian_ik import site_pose
from collect_act_grasp_data import GRIPPER_OPEN,HOME_Q,LEFT_HOME_Q,LEFT_ARM,RIGHT_ARM,MODEL_DIR,MAX_TIMESTEPS,reset_episode_state,save_episode,pad_contact_summary
from vr_left_grasp_scene import BASKET_BASE_Z,BASKET_DEPTH,BASKET_INNER_HALF,BASKET_WALL,BASKET_X,BASKET_Y,CUP_HALF,GRASP_HOLD_SECONDS,BASKET_HOLD_SECONDS,CUP_GRID_SIZE,cup_grid_xy,scene_xml
LF=["upoo_left_openarm_v1_finger_joint1","upoo_left_openarm_v1_finger_joint2"];RF=["upoo_right_openarm_v1_finger_joint1","upoo_right_openarm_v1_finger_joint2"];PADS=["upoo_left_finger1_inner_pad","upoo_left_finger2_inner_pad"]
HOME_POSE_FILE=WS/"src/kio_teleop_openarm/config/home_pose.yaml"
# Start from neutral redundancy resolution. Joint preferences are useful only
# after the Cartesian control frame and reachable workspace are established.
DEFAULT_JOINT_WEIGHTS=(1.,1.,1.,1.,1.,1.)
with open(HOME_POSE_FILE, encoding="utf-8") as _home_file:
 _home_pose=yaml.safe_load(_home_file)["home_pose"]
LEFT_HOME_Q=np.array([_home_pose["upoo_left_Base_J01"],_home_pose["upoo_left_J02"],_home_pose["upoo_left_J03"],_home_pose["upoo_left_J04"],_home_pose["upoo_left_J05"],_home_pose["upoo_left_J06"]],dtype=float)
_left_home_fingers=np.array([_home_pose["upoo_left_finger_right_joint"],_home_pose["upoo_left_finger_left_joint"]],dtype=float)
if not np.isfinite(_left_home_fingers).all()or not np.allclose(_left_home_fingers,_left_home_fingers[0])or not 0<=_left_home_fingers[0]<=GRIPPER_OPEN:raise ValueError("Left Home gripper fingers must be equal and in [0, GRIPPER_OPEN]")
LEFT_HOME_GRIPPER=float(_left_home_fingers[0])
def load_failure_targets(path):
 source=Path(path).expanduser().resolve();targets=[];seen=set()
 if not source.is_file():raise FileNotFoundError(f"Failure rollout file not found: {source}")
 with source.open(encoding="utf-8")as stream:
  for line_number,line in enumerate(stream,1):
   if not line.strip():continue
   try:record=json.loads(line)
   except json.JSONDecodeError as exc:raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}")from exc
   success=record.get("task_success",record.get("outcome")=="task_success")
   if success:continue
   rollout_id=record.get("rollout_id")
   pose=np.asarray(record.get("initial_object_pose",[]),dtype=float)
   if not isinstance(rollout_id,int)or rollout_id in seen:raise ValueError(f"Invalid or duplicate rollout_id at {source}:{line_number}")
   if pose.shape!=(7,)or not np.isfinite(pose).all():raise ValueError(f"Expected a finite 7D initial_object_pose at {source}:{line_number}")
   seen.add(rollout_id);targets.append({"rollout_id":rollout_id,"outcome":str(record.get("outcome","evaluation_failure")),"pose":pose})
 if not targets:raise ValueError(f"No failed rollouts found in {source}")
 return source,targets
def completed_failure_rollouts(output_dir,source):
 completed=set()
 for path in Path(output_dir).glob("episode_*.hdf5"):
  try:
   with h5py.File(path,"r")as root:
    recorded_source=root.attrs.get("source_evaluation_file")
    if isinstance(recorded_source,bytes):recorded_source=recorded_source.decode()
    if recorded_source==str(source)and"source_evaluation_rollout_id"in root.attrs:completed.add(int(root.attrs["source_evaluation_rollout_id"]))
  except OSError:continue
 return completed
def save_episode(output_dir,episode_idx,buffers):
 t_actual=len(buffers["action"]);path=os.path.join(output_dir,f"episode_{episode_idx}.hdf5")
 with h5py.File(path,"w",rdcc_nbytes=1024**2*2)as root:
  root.attrs["sim"]=True;root.attrs["valid_length"]=t_actual;root.attrs["initial_object_pose"]=np.asarray(buffers["initial_object_pose"],dtype=np.float32)
  for name in("source_evaluation_file","source_evaluation_rollout_id","source_evaluation_outcome"):
   if name in buffers:root.attrs[name]=buffers[name]
  obs=root.create_group("observations");images=obs.create_group("images")
  for name in buffers["camera_names"]:
   frames=buffers["images"][name];height,width,channels=frames[0].shape;dataset=images.create_dataset(name,(t_actual,height,width,channels),dtype="uint8",chunks=(1,height,width,channels),compression="lzf")
   for t,frame in enumerate(frames):dataset[t]=frame
  obs.create_dataset("qpos",(MAX_TIMESTEPS,14),dtype="float32");obs.create_dataset("qvel",(MAX_TIMESTEPS,14),dtype="float32");root.create_dataset("action",(MAX_TIMESTEPS,14),dtype="float32")
  root["/action"][:t_actual]=np.asarray(buffers["action"],dtype=np.float32);root["/observations/qpos"][:t_actual]=np.asarray(buffers["qpos"],dtype=np.float32);root["/observations/qvel"][:t_actual]=np.asarray(buffers["qvel"],dtype=np.float32)
  for key,last in (("/action",buffers["action"][-1]),("/observations/qpos",buffers["qpos"][-1]),("/observations/qvel",buffers["qvel"][-1])):root[key][t_actual:]=last
 print(f"[record] Saved episode_{episode_idx}.hdf5 ({t_actual} frames; compressed {buffers['camera_names']} images)")
def rot(x):
 u,_,v=np.linalg.svd(x);r=u@v
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@v
 return r
def rotvec(r):
 r=rot(r);cosine=np.clip((np.trace(r)-1)/2,-1,1);angle=np.arccos(cosine);vee=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
 if angle<1e-6:return .5*vee
 if np.pi-angle<1e-5:
  axis=np.sqrt(np.maximum((np.diag(r)+1)/2,0));axis/=max(np.linalg.norm(axis),1e-9);return axis*angle
 return vee*angle/(2*np.sin(angle))
def rotation_angle_deg(r):
 return float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))
def rotation_from_rotvec(v):
 angle=np.linalg.norm(v)
 if angle<1e-9:return np.eye(3)
 axis=v/angle;skew=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
 return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
def rigid_pose(matrix,rotation_tolerance=.1):
 """Return a normalized proper rigid transform, or None for invalid tracking."""
 try:t=np.asarray(matrix,dtype=float)
 except (TypeError,ValueError):return None
 if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],(0.,0.,0.,1.),atol=1e-2):return None
 r=t[:3,:3];det=np.linalg.det(r)
 if det<=.5 or np.linalg.norm(r.T@r-np.eye(3),ord="fro")>rotation_tolerance:return None
 result=t.copy();result[:3,:3]=rot(r);result[3]=(0.,0.,0.,1.);return result
def clip_norm(v,maximum):
 n=np.linalg.norm(v)
 return v if n<=maximum else v*(maximum/n)
def bounded_weighted_dls(jacobian,velocity,weight_matrix,damping,lower,upper,preferred_velocity=None):
 """Solve a weighted DLS step while respecting per-joint velocity bounds."""
 q=np.zeros(jacobian.shape[1]);free=np.ones(q.size,dtype=bool);preferred=np.zeros_like(q)if preferred_velocity is None else np.asarray(preferred_velocity,dtype=float)
 for _ in range(q.size+1):
  indices=np.flatnonzero(free)
  if not indices.size:return q
  j=jacobian[:,indices];w=weight_matrix[np.ix_(indices,indices)]
  fixed=np.flatnonzero(~free);rhs=j.T@(velocity-jacobian@q)+damping*damping*(w@preferred[indices]-weight_matrix[np.ix_(indices,fixed)]@(q[fixed]-preferred[fixed]))
  try:proposal=np.linalg.solve(j.T@j+damping*damping*w,rhs)
  except np.linalg.LinAlgError:proposal=np.linalg.lstsq(j.T@j+damping*damping*w,rhs,rcond=None)[0]
  candidate=q.copy();candidate[indices]=proposal
  below=candidate[indices]<lower[indices];above=candidate[indices]>upper[indices]
  if not np.any(below|above):return candidate
  violation=np.maximum(lower[indices]-candidate[indices],candidate[indices]-upper[indices])
  local=int(np.argmax(violation));joint=indices[local]
  q[joint]=lower[joint] if candidate[joint]<lower[joint] else upper[joint]
  free[joint]=False
 return q
def landmarks_yup_to_vuer(landmarks):
 """Convert raw TeleVision landmark positions into Vuer's Z-up frame."""
 points=np.asarray(landmarks,dtype=float)
 return points@grd_yup2grd_zup[:3,:3].T+grd_yup2grd_zup[:3,3]
def pose(p,r):
 t=np.eye(4);t[:3,:3]=rot(r);t[:3,3]=p;return t
def invert_pose(t):
 r=t[:3,:3];result=np.eye(4);result[:3,:3]=r.T;result[:3,3]=-r.T@t[:3,3];return result
def target_pose_from_hand(hand_current,hand_reference,tcp_reference,position_scale=1.):
 """Map hand XYZ and relative 3-DOF orientation to a six-axis TCP target."""
 target_p=tcp_reference[:3,3]+position_scale*(hand_current[:3,3]-hand_reference[:3,3])
 target_r=rot(hand_current[:3,:3]@hand_reference[:3,:3].T@tcp_reference[:3,:3])
 return target_p,target_r
def setcam(c,p,l):
 d=l-p;n=np.linalg.norm(d)
 if n<1e-6:return
 d/=n;c.type=mujoco.mjtCamera.mjCAMERA_FREE;c.lookat[:]=l;c.distance=n;c.elevation=np.degrees(np.arcsin(d[2]));c.azimuth=np.degrees(np.arctan2(d[1],d[0]))
class Vuer:
 def __init__(s):s.h=np.eye(4);s.l=np.eye(4)
 def read(s,t):
  s.h=mat_update(s.h,t.head_matrix.copy());s.l=mat_update(s.l,t.left_hand.copy())
  return grd_yup2grd_zup@s.h@fast_mat_inv(grd_yup2grd_zup),grd_yup2grd_zup@s.l@fast_mat_inv(grd_yup2grd_zup)
class Collector:
 READY,REC,PENDING="READY","REC","PENDING"
 def __init__(s,a):
  s.a=a;os.makedirs(a.output_dir,exist_ok=True);e=[int(x.split("_")[1].split(".")[0])for x in os.listdir(a.output_dir)if x.startswith("episode_")and x.endswith(".hdf5")];s.ep=max(e)+1 if e else 0;s._cup_y=getattr(a,"cup_y",None);s.failure_source=None;s.failure_targets=None;s.completed_failure_ids=set();s.failure_index=None
  if getattr(a,"failure_rollouts",None):
   if s._cup_y is not None:raise ValueError("--failure-rollouts cannot be combined with --cup-y")
   s.failure_source,s.failure_targets=load_failure_targets(a.failure_rollouts);s.completed_failure_ids=completed_failure_rollouts(a.output_dir,s.failure_source);s.failure_index=next((i for i,target in enumerate(s.failure_targets)if target["rollout_id"]not in s.completed_failure_ids),None)
   if s.failure_index is None:raise RuntimeError(f"All {len(s.failure_targets)} failure targets have already been recorded in {a.output_dir}")
   print(f"[targets] Loaded {len(s.failure_targets)} failed rollouts from {s.failure_source}; {len(s.completed_failure_ids)} already recorded")
  s.tmp=tempfile.mkdtemp(prefix="vr_left_");x,y=s.current_cup_xy()
  with open(s.tmp+"/scene.xml","w")as f:f.write(scene_xml(x,y))
  if os.path.isdir(MODEL_DIR+"/assets"):os.symlink(MODEL_DIR+"/assets",s.tmp+"/assets")
  s.m=mujoco.MjModel.from_xml_path(s.tmp+"/scene.xml");s.d=mujoco.MjData(s.m);I=lambda k,n:mujoco.mj_name2id(s.m,k,n)
  def q(ns,vel=False):return[(s.m.jnt_dofadr if vel else s.m.jnt_qposadr)[I(mujoco.mjtObj.mjOBJ_JOINT,n)]for n in ns]
  s.lq,s.rq,s.lf,s.rf=q(LEFT_ARM),q(RIGHT_ARM),q(LF),q(RF);s.lv,s.rv=q(LEFT_ARM,1),q(RIGHT_ARM,1);s.lfv,s.rfv=q(LF,1)[0],q(RF,1)[0];s.left_limits=np.array([s.m.jnt_range[I(mujoco.mjtObj.mjOBJ_JOINT,n)]for n in LEFT_ARM])
  if not np.isfinite(a.home_joint_margin)or a.home_joint_margin<0 or np.any(s.left_limits[:,1]-s.left_limits[:,0]<=2*a.home_joint_margin):raise ValueError("--home-joint-margin must be non-negative and smaller than half of every arm joint range")
  s.home_q=np.clip(LEFT_HOME_Q,s.left_limits[:,0]+a.home_joint_margin,s.left_limits[:,1]-a.home_joint_margin)
  s.tcp,s.cup,s.cg=I(mujoco.mjtObj.mjOBJ_SITE,"upoo_left_tcp"),I(mujoco.mjtObj.mjOBJ_BODY,"cup"),I(mujoco.mjtObj.mjOBJ_GEOM,"cup_body");s.pads=[I(mujoco.mjtObj.mjOBJ_GEOM,x)for x in PADS];s.cqa=s.m.jnt_qposadr[s.m.body_jntadr[s.cup]]
  first_left_joint=I(mujoco.mjtObj.mjOBJ_JOINT,LEFT_ARM[0]);s.left_base=s.m.body_parentid[s.m.jnt_bodyid[first_left_joint]]
  def acts(p):return sorted([i for i in range(s.m.nu)if((mujoco.mj_id2name(s.m,mujoco.mjtObj.mjOBJ_ACTUATOR,i)or"").startswith(p)and(mujoco.mj_id2name(s.m,mujoco.mjtObj.mjOBJ_ACTUATOR,i)or"").endswith("_ctrl"))],key=lambda i:mujoco.mj_id2name(s.m,mujoco.mjtObj.mjOBJ_ACTUATOR,i))
  s.la,s.ra=acts("upoo_left_J"),acts("upoo_right_J");s.lg,s.rg=I(mujoco.mjtObj.mjOBJ_ACTUATOR,"upoo_left_gripper_ctrl"),I(mujoco.mjtObj.mjOBJ_ACTUATOR,"upoo_right_gripper_ctrl")
  if min(s.tcp,s.cup,s.cg,*s.pads,s.lg,s.rg)<0 or len(s.la)!=6 or len(s.ra)!=6:raise RuntimeError("required left-arm model objects are missing")
  s.ren={n:mujoco.Renderer(s.m,height=a.image_height,width=a.image_width)for n in("vr_center","wrist")};s.video_renderer=mujoco.Renderer(s.m,height=a.image_height,width=a.image_width);s.recording=False;s.video_writer=None;s.video_next_frame=0.;s.state=s.READY;s.buf=None;s.cal=False;s.manual_panel=False;s.q=s.home_q.copy();s.g=GRIPPER_OPEN;s.t=0.;s.last_teleop=None;s.wrist_frame=None;s.teleop_log=0.;s.last_q_velocity=np.zeros(6);s.reset()
  s.joint_weights=np.ones(6)if a.joint_weights is None else np.asarray(a.joint_weights,dtype=float)
  if s.joint_weights.shape!=(6,)or not np.isfinite(s.joint_weights).all()or np.any(s.joint_weights<=0):raise ValueError("--joint-weights must contain six finite values greater than zero")
  s.joint_weight_matrix=np.diag(s.joint_weights*s.joint_weights)
 def current_failure_target(s):
  return None if s.failure_targets is None else s.failure_targets[s.failure_index]
 def current_cup_xy(s):
  target=s.current_failure_target()
  if target is not None:return float(target["pose"][0]),float(target["pose"][1])
  x,y=cup_grid_xy(s.ep);return x,s._cup_y if s._cup_y is not None else y
 def advance_failure_target(s):
  target=s.current_failure_target()
  if target is None:return False
  s.completed_failure_ids.add(target["rollout_id"]);s.failure_index=next((i for i,item in enumerate(s.failure_targets)if item["rollout_id"]not in s.completed_failure_ids),None)
  return s.failure_index is None
 def controls(s,force=False):
  # Leave left-arm actuator targets editable from the MuJoCo control panel
  # until VR calibration, or whenever manual-panel mode is enabled.
  if force or (s.cal and not s.manual_panel):
   for i,a in enumerate(s.la):s.d.ctrl[a]=s.q[i]
  for i,a in enumerate(s.ra):s.d.ctrl[a]=HOME_Q[i]
  s.d.ctrl[s.lg]=s.g;s.d.ctrl[s.rg]=GRIPPER_OPEN
 def reset(s):
  x,y=s.current_cup_xy();reset_episode_state(s.m,s.d,s.rq,s.lq,s.rf,s.cup,s.cqa,x,y)
  for i,q in enumerate(s.lq):s.d.qpos[q]=s.home_q[i]
  for i in s.lf+s.rf:s.d.qpos[i]=GRIPPER_OPEN
  s.q=s.home_q.copy();s.g=GRIPPER_OPEN;s.last_teleop=None;s.hand_ref=None;s.eref=None;s.last_q_velocity=np.zeros(6);s.manual_panel=False;mujoco.mj_forward(s.m,s.d);s.controls(force=True)
  for _ in range(int(1/s.m.opt.timestep)):mujoco.mj_step(s.m,s.d)
  s.t=0.;s.grasp_since=None;s.grasped=False;s.basket_since=None;s.cal=False;s.calibrate_at=None
  target=s.current_failure_target();target_text="" if target is None else f"; failure target {s.failure_index+1}/{len(s.failure_targets)}, rollout {target['rollout_id']} ({target['outcome']})"
  print(f"[scene] Cup at ({x:.3f},{y:.3f}); basket at ({BASKET_X:.3f},{BASKET_Y:.3f}){target_text}; calibration required")
 def connect(s):
  from multiprocessing import Event,Queue,shared_memory
  s.sh=shared_memory.SharedMemory(create=True,size=480*1280*3);s.img=np.ndarray((480,1280,3),np.uint8,s.sh.buf);s.img.fill(0);s.tv=OpenTeleVision((480,640),s.sh.name,Queue(),Event(),ngrok=s.a.ngrok,cert_file=s.a.cert_file,key_file=s.a.key_file);s.v=Vuer()
  s.gl=mujoco.GLContext(640,480);s.gl.make_current();s.sc=mujoco.MjvScene(s.m,maxgeom=10000);s.op=mujoco.MjvOption();s.cl,s.cr=mujoco.MjvCamera(),mujoco.MjvCamera();s.rl=mujoco.MjrContext(s.m,mujoco.mjtFontScale.mjFONTSCALE_150.value);s.rr=mujoco.MjrContext(s.m,mujoco.mjtFontScale.mjFONTSCALE_150.value);s.vp=mujoco.MjrRect(0,0,640,480);s.hp=np.array([.12,0,1.08]);s.f=np.array([-.38,0,.48])-s.hp;s.f/=np.linalg.norm(s.f);s.head_left=np.cross([0,0,1],s.f);s.head_left/=np.linalg.norm(s.head_left);s.head_rot=np.column_stack((s.f,s.head_left,np.cross(s.f,s.head_left)));s.world_head=pose(s.hp,s.head_rot);s.off=(s.head_left*.032,-s.head_left*.032);s.log=0.;print("[vr] MuJoCo stereo stream active")
 def render(s,h):
  d=np.eye(3)if not s.cal else rot((s.world_vuer@h)[:3,:3]@s.head_rot.T);s.gl.make_current()
  for c,o,ctx,out in((s.cl,s.off[0],s.rl,s.img[:,:640]),(s.cr,s.off[1],s.rr,s.img[:,640:])):
   p=s.hp+d@o;setcam(c,p,p+d@s.f);mujoco.mjv_updateScene(s.m,s.d,s.op,None,c,mujoco.mjtCatBit.mjCAT_ALL,s.sc);mujoco.mjr_render(s.vp,s.sc,ctx);im=np.empty((480,640,3),np.uint8);mujoco.mjr_readPixels(im,None,s.vp,ctx);out[:]=im[::-1]
  if s.a.show_wrist:
   s.ren["wrist"].update_scene(s.d,camera="wrist");cv2.imshow("Wrist RGB",cv2.cvtColor(s.ren["wrist"].render(),cv2.COLOR_RGB2BGR));cv2.waitKey(1)
  if time.perf_counter()-s.log>5:s.log=time.perf_counter();print(f"[vr] frame max={s.img.max()} nonzero={s.img.any(axis=2).mean()*100:.1f}%")
 def calibrate(s):
  s.calibrate_at=time.perf_counter();print("[calib] Capturing now; keep your head and left hand steady.")
 def body_pose(s,body_id):
  return pose(s.d.xpos[body_id].copy(),s.d.xmat[body_id].reshape(3,3).copy())
 def landmarks_in_vuer(s):
  try:lm=np.asarray(s.tv.left_landmarks,dtype=float).reshape(-1,3)
  except (AttributeError,TypeError,ValueError):return None
  if lm.shape[0]<9 or not np.isfinite(lm).all() or not np.any(lm):return None
  return landmarks_yup_to_vuer(lm)
 def current_hand_geometry(s):
  """Return thumb-index pinch distance in meters, solely for gripper control."""
  landmarks=s.landmarks_in_vuer()
  if landmarks is None or landmarks.shape[0]<=9:return None
  thumb,index=landmarks[4],landmarks[9]
  if not np.isfinite((thumb,index)).all():return None
  return float(np.linalg.norm(index-thumb))
 def update_gripper(s,dt,pinch_distance):
  if pinch_distance is None:return
  target=np.clip(pinch_distance/s.a.gripper_open_distance,0,1)*GRIPPER_OPEN
  alpha=1-np.exp(-dt/s.a.gripper_smoothing_tau);smoothed=s.g+(target-s.g)*alpha;max_step=s.a.gripper_max_speed*dt;s.g=np.clip(s.g+np.clip(smoothed-s.g,-max_step,max_step),0,GRIPPER_OPEN)
 def finish_calibration(s,h,l):
  if getattr(s,"calibrate_at",None) is None or time.perf_counter()<s.calibrate_at:return
  h,l=rigid_pose(h),rigid_pose(l)
  if h is None or l is None:
   s.calibrate_at=None;print("[calib] Head or left-hand matrix unavailable; hold the hand visible and press P again.");return
  s.calibrate_at=None
  # The calibration establishes a VR-relative frame, not a fixed target point.
  s.world_robotbase=s.body_pose(s.left_base);s.robotbase_world=invert_pose(s.world_robotbase)
  s.world_vuer=s.world_head@invert_pose(h);s.robotbase_vuer=s.robotbase_world@s.world_vuer
  s.hand_ref=s.robotbase_vuer@l.copy()
  p,r=site_pose(s.d,s.tcp);s.eref=s.robotbase_world@pose(p,r)
  s.last_teleop=time.perf_counter();s.q=s.d.qpos[s.lq].copy();s.last_q_velocity=np.zeros(6);s.manual_panel=False;s.cal=True
  print("[calib] Ready: wrist-matrix relative 6D control active.")
 def teleop(s,l):
  if not s.cal:return
  now=time.perf_counter();dt=np.clip(now-s.last_teleop if s.last_teleop else 1/s.a.control_hz,1e-4,.1);s.last_teleop=now
  s.update_gripper(dt,s.current_hand_geometry())
  l=rigid_pose(l)
  if l is None:return
  hand_current=s.robotbase_vuer@l.copy();hand_current[:3,:3]=rot(hand_current[:3,:3])
  target_p,target_r=target_pose_from_hand(hand_current,s.hand_ref,s.eref,s.a.position_scale)
  s.incremental_ik(target_p,target_r,dt)
  if s.a.teleop_debug and now-s.teleop_log>=.5:
   s.teleop_log=now;current_p,current_r=site_pose(s.d,s.tcp);world_target_p=s.world_robotbase[:3,:3]@target_p+s.world_robotbase[:3,3];world_target_r=s.world_robotbase[:3,:3]@target_r
   print("[teleop] wrist_delta="+np.array2string(hand_current[:3,3]-s.hand_ref[:3,3],precision=3,formatter={"float":lambda x:f"{x:+.4f}"})+" wrist_rot_delta="+f"{rotation_angle_deg(hand_current[:3,:3]@s.hand_ref[:3,:3].T):.1f}deg"+" pos_err="+f"{np.linalg.norm(world_target_p-current_p):.3f}m"+" ori_err="+f"{rotation_angle_deg(world_target_r@current_r.T):.1f}deg"+" q_vel="+np.array2string(s.last_q_velocity,precision=2))
 def incremental_ik(s,target_p_base,target_r_base,dt):
  # Solve from the persistent six-joint command. Solving from lagging actuator
  # qpos every frame discards most wrist progress before the servo catches up.
  q_actual=s.d.qpos[s.lq].copy();s.d.qpos[s.lq]=s.q;mujoco.mj_forward(s.m,s.d);current_p,current_r=site_pose(s.d,s.tcp)
  target_p=s.world_robotbase[:3,:3]@target_p_base+s.world_robotbase[:3,3];target_r=s.world_robotbase[:3,:3]@target_r_base
  position_error=target_p-current_p;rotation_error=rotvec(target_r@current_r.T)
  jac_p=np.zeros((3,s.m.nv));jac_r=np.zeros((3,s.m.nv));mujoco.mj_jacSite(s.m,s.d,jac_p,jac_r,s.tcp);jacobian=np.vstack((jac_p[:,s.lv],jac_r[:,s.lv]))
  s.d.qpos[s.lq]=q_actual;mujoco.mj_forward(s.m,s.d)
  linear_velocity=clip_norm(s.a.tcp_linear_gain*position_error,s.a.tcp_max_linear_speed)
  angular_velocity=clip_norm(s.a.tcp_angular_gain*rotation_error,s.a.tcp_max_angular_speed)
  speed_limit=np.full(6,s.a.joint_max_speed);acceleration_limit=s.a.joint_max_acceleration*dt
  margin=s.a.joint_limit_margin;lower_limit=s.left_limits[:,0]+margin;upper_limit=s.left_limits[:,1]-margin
  lower=np.maximum.reduce((-speed_limit,s.last_q_velocity-acceleration_limit,(lower_limit-s.q)/dt));upper=np.minimum.reduce((speed_limit,s.last_q_velocity+acceleration_limit,(upper_limit-s.q)/dt))
  dq=bounded_weighted_dls(jacobian,np.r_[linear_velocity,angular_velocity],s.joint_weight_matrix,s.a.ik_damping,lower,upper)
  q_previous=s.q.copy();s.q=np.clip(q_previous+dq*dt,lower_limit,upper_limit);s.last_q_velocity=(s.q-q_previous)/dt;return True
 def start(s):
  if not s.cal:print("[record] calibrate first with P");return
  s.state=s.REC;s.next=s.t;s.grasp_since=None;s.grasped=False;s.basket_since=None;s.buf={"action":[],"qpos":[],"qvel":[],"images":{"vr_center":[],"wrist":[]},"camera_names":["vr_center","wrist"],"initial_object_pose":s.d.qpos[s.cqa:s.cqa+7].copy()}
  target=s.current_failure_target()
  if target is not None:s.buf.update({"source_evaluation_file":str(s.failure_source),"source_evaluation_rollout_id":target["rollout_id"],"source_evaluation_outcome":target["outcome"]})
  print("[record] START (vr_center + wrist RGB)")
 def in_basket(s):
  p=s.d.xpos[s.cup];half=BASKET_INNER_HALF-CUP_HALF;floor=BASKET_BASE_Z+BASKET_WALL+CUP_HALF-.002;rim=BASKET_BASE_Z+BASKET_DEPTH
  return abs(p[0]-BASKET_X)<=half and abs(p[1]-BASKET_Y)<=half and floor<=p[2]<=rim
 def frame(s):
  if s.state!=s.REC or s.t<s.next:return False
  s.next+=1/s.a.record_fps;b=s.buf;b["action"].append(np.r_[s.q,s.g/GRIPPER_OPEN,HOME_Q,1.]);b["qpos"].append(np.r_[[s.d.qpos[x]for x in s.lq],s.d.qpos[s.lf[0]]/GRIPPER_OPEN,[s.d.qpos[x]for x in s.rq],s.d.qpos[s.rf[0]]/GRIPPER_OPEN]);b["qvel"].append(np.r_[[s.d.qvel[x]for x in s.lv],s.d.qvel[s.lfv]/GRIPPER_OPEN,[s.d.qvel[x]for x in s.rv],s.d.qvel[s.rfv]/GRIPPER_OPEN])
  for n,r in s.ren.items():
   r.update_scene(s.d,camera=n);image=r.render().copy();b["images"][n].append(image)
   if n=="wrist":s.wrist_frame=image
  c,_=pad_contact_summary(s.m,s.d,s.cg,s.pads);bilateral=c[0]>0 and c[1]>0
  if bilateral:
   s.grasp_since=s.t if s.grasp_since is None else s.grasp_since
   if not s.grasped and s.t-s.grasp_since>=GRASP_HOLD_SECONDS:s.grasped=True;print("[record] grasp acquired")
  else:s.grasp_since=None
  in_basket=s.grasped and s.in_basket();s.basket_since=s.t if in_basket and s.basket_since is None else(s.basket_since if in_basket else None)
  if s.basket_since is not None and s.t-s.basket_since>=BASKET_HOLD_SECONDS:
   save_episode(s.a.output_dir,s.ep,b);s.ep+=1;s.buf=None;s.state=s.READY;print("[record] SUCCESS saved; calibration cleared")
   if s.advance_failure_target():print(f"[targets] COMPLETE: recorded all {len(s.failure_targets)} failed rollout positions");return True
   s.reset()
  elif len(b["action"])>=MAX_TIMESTEPS:s.state=s.PENDING;print("[record] limit reached, press D")
  return False
 def run(s):
  print("MuJoCo VR ACT Left-Arm: Space start/stop, D discard/reset, O reset, P calibrate, M panel-control, Q/Esc quit");print("[panel] Before P, use the MuJoCo control panel to pose the left arm; press P when ready.");s.connect();keys=[];stop=threading.Event();old=termios.tcgetattr(0)if sys.stdin.isatty()else None
  if old:tty.setcbreak(0)
  def reader():
   while not stop.is_set():
    r,_,_=select.select([sys.stdin],[],[],.1)
    if r:keys.append(sys.stdin.read(1).lower())
  threading.Thread(target=reader,daemon=True).start();run=True;wall=time.perf_counter();s.next_control=wall
  try:
   with mujoco.viewer.launch_passive(s.m,s.d)as viewer:
    while run and viewer.is_running():
     h,l=s.v.read(s.tv)
     while keys:
      k=keys.pop(0)
      if k==" ":
       if s.state==s.READY:s.start()
       elif s.state==s.REC:s.state=s.PENDING;print("[record] stopped, press D")
      elif k=="d"and s.state!=s.REC:s.buf=None;s.state=s.READY;s.reset()
      elif k=="o":
       if s.state==s.REC:print("[scene] O disabled while recording")
       else:s.buf=None;s.state=s.READY;s.reset()
      elif k=="p":s.calibrate()
      elif k=="m":
       s.manual_panel=not s.manual_panel
       if s.manual_panel:print("[panel] Manual control ON: left-arm targets are no longer overwritten by VR.")
       else:
        s.q=s.d.qpos[s.lq].copy();s.last_q_velocity=np.zeros(6);print("[panel] Manual control OFF: VR control resumed from the current simulated pose.")
      elif k=="v":
       if not s.recording:
        p=os.path.join(s.a.output_dir,f"video_{s.ep}_{int(time.time())}.mp4")
        s.video_writer=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*"mp4v"),30.0,(s.a.image_width,s.a.image_height))
        if s.video_writer.isOpened():s.recording=True;s.video_next_frame=s.t;print(f"[video] REC: {p}")
        else:print("[video] cant open",p);s.video_writer=None
       else:s.video_writer.release();s.video_writer=None;s.recording=False;print("[video] stopped")
      elif k in("q","\x1b"):run=False
     s.finish_calibration(h,l);now=time.perf_counter()
     if now>=s.next_control:
      s.teleop(l);s.next_control+=max(1,int((now-s.next_control)*s.a.control_hz)+1)/s.a.control_hz
     s.controls();n=min(50,max(1,int((now-wall)/s.m.opt.timestep)));wall=now
     for _ in range(n):mujoco.mj_step(s.m,s.d)
     s.t+=n*s.m.opt.timestep;targets_complete=s.frame();s.render(h);
     if targets_complete:run=False
     if s.recording and s.video_writer is not None and s.t>=s.video_next_frame:
      s.video_renderer.update_scene(s.d,camera="vr_center")
      s.video_writer.write(cv2.cvtColor(s.video_renderer.render(),cv2.COLOR_RGB2BGR))
      s.video_next_frame+=1./30.
     viewer.sync();time.sleep(.001)
  finally:
   stop.set()
   if old:termios.tcsetattr(0,termios.TCSADRAIN,old)
   for r in s.ren.values():r.close()
   if s.video_writer is not None:s.video_writer.release()
   if s.a.show_wrist:cv2.destroyWindow("Wrist RGB")

   s.rl.free();s.rr.free();s.gl.free();s.sh.close();s.sh.unlink();shutil.rmtree(s.tmp,ignore_errors=True)
def parse():
 p=argparse.ArgumentParser();p.add_argument("--home-joint-margin",type=float,default=0.0,help="radians kept inside each arm joint limit at reset; default: 0.3");p.add_argument("--output-dir",default=str(WS/"data"/"video"));p.add_argument("--show-wrist",action=argparse.BooleanOptionalAction,default=True,help="show a live Wrist RGB preview window");p.add_argument("--position-scale",type=float,default=1.,help="relative pinch-centre to TCP displacement scale");p.add_argument("--record-fps",type=int,default=50);p.add_argument("--control-hz",type=float,default=60.);p.add_argument("--image-width",type=int,default=640);p.add_argument("--image-height",type=int,default=480);p.add_argument("--tcp-max-linear-speed",type=float,default=.6);p.add_argument("--tcp-max-angular-speed",type=float,default=2.5);p.add_argument("--tcp-linear-gain",type=float,default=5.);p.add_argument("--tcp-angular-gain",type=float,default=2.);p.add_argument("--joint-weights",type=float,nargs=6,default=DEFAULT_JOINT_WEIGHTS,metavar=("W1","W2","W3","W4","W5","W6"),help="per-joint DLS damping weights; higher means less movement; order J01 J02 J03 J04 J05 J06; default: all 1.0");p.add_argument("--ik-damping",type=float,default=.08);p.add_argument("--joint-max-speed",type=float,default=2.0);p.add_argument("--joint-max-acceleration",type=float,default=10.);p.add_argument("--joint-limit-margin",type=float,default=0.);p.add_argument("--gripper-open-distance",type=float,default=.15,help="meters; thumb-index distance for fully open gripper");p.add_argument("--gripper-smoothing-tau",type=float,default=.08);p.add_argument("--gripper-max-speed",type=float,default=.12);p.add_argument("--cup-y",type=float,default=None,help="override cup Y coordinate (X still cycles through grid)");p.add_argument("--failure-rollouts",type=Path,help="evaluation rollouts.jsonl; record each failed initial pose in rollout order");p.add_argument("--teleop-debug",action="store_true",help="print hand-frame, TCP, and IK diagnostics");p.add_argument("--ngrok",action="store_true");p.add_argument("--cert-file",default=str(D/"192.168.0.5+2.pem"));p.add_argument("--key-file",default=str(D/"192.168.0.5+2-key.pem"));return p.parse_args()
if __name__=="__main__":Collector(parse()).run()
