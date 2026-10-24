#!/usr/bin/env python3
"""Pi ONNX host accepting the --model file; fixed original 7-feature ABI and action pipeline."""
import argparse
import hashlib
import math
from pathlib import Path
import struct
import time
import zlib

import numpy as np

ROOT=Path(__file__).resolve().parent
# Model identity embedded here; no external JSON file is needed.
MODEL_IDENTITY={'policy_id': 'rl_mlp:08d2fc39e888', 'config_id': '297ed5ad0ee0', 'model_sha256': '8ca2b1cf7e2bef7757c8026d77b6aa5a89ac3c2a85b9ce7ddaa4e65f476da271', 'source_header_sha256': 'fb89f4dc41b3d233a63b15bc66db0b95f01bcfcd1b91f1492d46e2ea48ddeda2'}
OBS=struct.Struct("<II7f4BfIII")
PACKET=struct.Struct("<4sII")
MAGICS=(b"LFO2",b"LFS2",b"LFI2")
POLICY_HZ=60.0
MAX_LINE_LOSS_S=0.5
STATES=("ACCEPTED","FRAME_SHAPE","EMPTY_HISTOGRAM","SINGLE_POPULATION",
        "CONTRAST_GUARD","DARK_FRACTION_GUARD","BRIGHT_FRACTION_GUARD",
        "CENTERLINE_GAP","FIT_FAILED")

def f32(v):
    return float(np.float32(v))

def minimum_pass(left,right):
    difference=f32(abs(f32(left-right)))
    common=f32(abs(f32(left+right)))
    severity=f32(difference/max(f32(difference+common),1e-9))
    minimum=f32(f32(0.64)+f32(f32(0.26)*severity))
    lmag,rmag=abs(left),abs(right)
    if f32(left*right)>0 and min(lmag,rmag)>1e-9:
        offset=f32(max(0.0,f32(minimum-min(lmag,rmag))))
        left=f32(left+math.copysign(offset,left))
        right=f32(right+math.copysign(offset,right))
    else:
        if lmag>1e-9:left=f32(math.copysign(max(lmag,minimum),left))
        if rmag>1e-9:right=f32(math.copysign(max(rmag,minimum),right))
    return float(np.clip(left,-1,1)),float(np.clip(right,-1,1))

def policy_step(raw,observation):
    """Port of lf_policy_step in the working header, not a new controller."""
    left,right=map(float,np.tanh(np.asarray(raw,dtype=np.float32)))
    if observation[2]<=0:
        left=right=0.0  # LF_BLIND_DUTY_SCALE=0.
    # Original release threshold, two loaded passes, then 0.2/tick slew.
    left=left if abs(left)>=f32(0.32) else 0.0
    right=right if abs(right)>=f32(0.32) else 0.0
    left,right=minimum_pass(left,right)
    left,right=minimum_pass(left,right)
    slew=f32(12.0/POLICY_HZ)
    prev_l,prev_r=map(float,observation[5:7])
    left=float(np.clip(left,f32(prev_l-slew),f32(prev_l+slew)))
    right=float(np.clip(right,f32(prev_r-slew),f32(prev_r+slew)))
    return np.asarray([left,right],dtype=np.float32)

class Policy:
    def __init__(self,model):
        import onnxruntime as ort
        identity=MODEL_IDENTITY
        actual=hashlib.sha256(Path(model).read_bytes()).hexdigest()
        same_actor=actual==identity["model_sha256"]
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),options,providers=["CPUExecutionProvider"])
        inputs,outputs=self.session.get_inputs(),self.session.get_outputs()
        if len(inputs)!=1 or inputs[0].shape!=[1,7] or inputs[0].type!="tensor(float)":
            raise ValueError("Expected float32 [1,7] observation")
        if len(outputs)!=1 or outputs[0].shape!=[1,2] or outputs[0].type!="tensor(float)":
            raise ValueError("Expected [1,2] raw actor output")
        self.input,self.output=inputs[0].name,outputs[0].name
        self.session.run([self.output],{self.input:np.zeros((1,7),dtype=np.float32)})
        if same_actor:
            print(f'MATCHED original actor: {identity["policy_id"]}',flush=True)
        else:
            print(f'SELECTED actor: {model}; SHA256={actual[:12]} (different trained weights accepted)',flush=True)
            print('Assumed ABI: [e_y, angle_rad, confidence, RAW rpm_left, RAW rpm_right, prev_left, prev_right].',flush=True)
            print('Assumed output: RAW actor logits; host applies tanh and original loaded-duty/slew pipeline.',flush=True)
        print(f'Required MCU configuration: {identity["config_id"]}',flush=True)

    def infer(self,observation):
        x=np.asarray(observation,dtype=np.float32).reshape(1,7)
        if not np.isfinite(x).all():raise ValueError("Nonfinite observation")
        # RAW signed RPM, as in lf_policy_target. No /420 normalization.
        raw=np.asarray(self.session.run([self.output],{self.input:x})[0],dtype=np.float32).reshape(2)
        if not np.isfinite(raw).all():raise ValueError("Nonfinite actor")
        return raw,policy_step(raw,x[0])

class Reader:
    def __init__(self):
        self.buffer=bytearray()
        self.bad_crc=0
        self.bytes_received=0

    def feed(self,data):
        self.bytes_received+=len(data)
        self.buffer.extend(data)
        result=[]
        while self.buffer:
            positions=[p for p in (self.buffer.find(m) for m in MAGICS) if p>=0]
            if not positions:
                self.buffer=self.buffer[-3:]
                break
            start=min(positions)
            if start:del self.buffer[:start]
            if len(self.buffer)<PACKET.size:break
            magic,length,crc=PACKET.unpack_from(self.buffer)
            if (magic==b"LFO2" and length!=OBS.size) or (magic==b"LFI2" and length!=76800) or (magic==b"LFS2" and length>4096):
                del self.buffer[0];continue
            end=PACKET.size+length
            if len(self.buffer)<end:break
            payload=bytes(self.buffer[PACKET.size:end])
            if zlib.crc32(payload)&0xffffffff!=crc:
                self.bad_crc+=1;del self.buffer[0];continue
            del self.buffer[:end]
            result.append((magic,payload))
        return result

def save_png(path,image):
    def chunk(kind,data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xffffffff)
    height,width=image.shape
    pixels=b"".join(b"\x00"+row.tobytes() for row in image)
    data=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,0,0,0,0))
    data+=chunk(b"IDAT",zlib.compress(pixels))+chunk(b"IEND",b"")
    Path(path).write_bytes(data)

def send(port,text):
    data=text.encode("ascii")
    if port.write(data)!=len(data):raise IOError("Short command write")

def decode_observation(payload):
    fields=OBS.unpack(payload)
    return {
        "seq":fields[0],"timestamp_us":fields[1],
        "observation":np.asarray(fields[2:9],dtype=np.float32),
        "valid":fields[9],"state":fields[10],"threshold":fields[11],"points":fields[12],
        "loss_s":fields[13],"camera_age_us":fields[14],"flags":fields[15],"accepted":fields[16]
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default=str(ROOT/"policy_esp32.onnx"))
    p.add_argument("--port",default="/dev/ttyACM0")
    p.add_argument("--drive",action="store_true")
    p.add_argument("--camera-only",action="store_true",help="Monitor MCU perception/encoder without ONNX")
    p.add_argument("--snapshot",metavar="PATH",help="Save one raw 320x240 camera PNG; motors stopped")
    p.add_argument("--check",action="store_true",help="Model and action pipeline only; no USB")
    args=p.parse_args()
    if args.drive and (args.camera_only or args.snapshot):
        p.error("--drive cannot combine with --camera-only or --snapshot")
    if args.check and (args.snapshot or args.camera_only):
        p.error("--check cannot combine with --camera-only or --snapshot")
    policy=None if args.camera_only or args.snapshot else Policy(args.model)
    if args.check:
        for obs in ([0,0,1,0,0,0,0],[.2,.1,1,100,100,.7,.7],[-.2,-.1,1,-100,100,-.7,.7]):
            raw,duty=policy.infer(obs)
            print("obs=",obs,"raw=",raw.tolist(),"postprocessed duty=",duty.tolist())
        print("Model/pipeline OK; test hardware and timing separately.")
        return
    import serial
    port=None
    try:
        port=serial.Serial(args.port,921600,timeout=.002,write_timeout=.05,exclusive=True)
        port.dtr=True;port.rts=False
        time.sleep(.2)
        send(port,"S\n");time.sleep(.05);port.reset_input_buffer()
        # Verify MCU identity in monitor mode before arming drive.
        send(port,"Q\n" if args.snapshot else "M\n")
        print("SNAPSHOT" if args.snapshot else "DRIVE" if args.drive else "MONITOR",args.port,flush=True)
        reader=Reader();started=time.monotonic();last_data=started;last_log=0.;last_seq=None
        count=0;rate_start=started;link_verified=False;host_armed=False
        while True:
            packets=reader.feed(port.read(min(max(port.in_waiting,1),65536)))
            latest=None
            for magic,payload in packets:
                if magic==b"LFS2":
                    status=payload.decode("utf-8","replace")
                    print("ESP32:",status,flush=True)
                    if f"policy={MODEL_IDENTITY['policy_id']}" not in status or f"config={MODEL_IDENTITY['config_id']}" not in status:
                        raise ValueError("ESP32 config differs from this matched package")
                    link_verified=True
                    if args.drive and not host_armed:
                        send(port,"A\n");host_armed=True
                        print("Matched ESP32 identity; drive armed.",flush=True)
                elif magic==b"LFI2" and args.snapshot:
                    save_png(args.snapshot,np.frombuffer(payload,np.uint8).reshape(240,320))
                    print("Saved",args.snapshot,flush=True)
                    return
                elif magic==b"LFO2":
                    latest=decode_observation(payload)
                    latest["received_at"]=time.monotonic()
                    count+=1
            now=time.monotonic()
            if latest is not None and latest["seq"]!=last_seq:
                last_seq=latest["seq"];last_data=now
                x=latest["observation"]
                if not np.isfinite(x).all():raise ValueError("Invalid observation")
                can_run=(latest["flags"]&3)==3 and latest["loss_s"]<=MAX_LINE_LOSS_S and latest["camera_age_us"]<=150000
                raw=np.zeros(2,np.float32);duty=np.zeros(2,np.float32)
                if policy is not None and can_run:
                    raw,duty=policy.infer(x)
                processing_ms=(time.monotonic()-latest["received_at"])*1000
                command=duty if args.drive and link_verified and host_armed and (latest["flags"]&4) and can_run and processing_ms<50 else np.zeros(2,np.float32)
                send(port,f'D {latest["seq"]} {command[0]:.8f} {command[1]:.8f}\n')
                if now-last_log>=1:
                    state=STATES[latest["state"]] if latest["state"]<len(STATES) else "UNKNOWN"
                    hz=count/max(now-rate_start,.001)
                    print(f'obs={latest["seq"]} hz={hz:.1f} detect={state} points={latest["points"]} '
                          f'ey={x[0]:+.3f} angle={x[1]:+.3f} conf={x[2]:.2f} '
                          f'rpm=({x[3]:+.1f},{x[4]:+.1f}) prev=({x[5]:+.3f},{x[6]:+.3f}) '
                          f'raw=({raw[0]:+.3f},{raw[1]:+.3f}) sent=({command[0]:+.3f},{command[1]:+.3f}) '
                          f'camera_age={latest["camera_age_us"]/1000:.1f}ms process={processing_ms:.1f}ms CRC={reader.bad_crc}',
                          flush=True)
                    last_log=now;count=0;rate_start=now
            if args.snapshot:
                if now-started>10:raise TimeoutError("No snapshot in 10 seconds; inspect MCU status")
            elif now-last_data>.15:
                # Stop on loss of telemetry, then require an explicit restart.
                raise TimeoutError("No observation for 150 ms; MCU watchdog stops motors")
    except KeyboardInterrupt:
        print("\nStopped.",flush=True)
    finally:
        if port is not None:
            try:send(port,"S\n")
            except Exception:pass
            port.close()

if __name__=="__main__":
    main()
