#!/usr/bin/env python3
"""Standalone ONNX line-following host for Raspberry Pi 4 + ESP32-S3.
Dependencies: numpy onnxruntime pyserial (optional: opencv-python for --preview).
Synchronized with esp32_pi_bridge.ino (G / S / D sequence left right).
Monitor: python3 control_synced.py --model policy.onnx --port /dev/ttyACM0
Drive: python3 control_synced.py --model policy.onnx --port /dev/ttyACM0 --drive
Camera only: python3 control_synced.py --camera-only --port /dev/ttyACM0
Check: python3 control_synced.py --model policy.onnx --check
CRC fix: 160x120 USB frames upsampled to 320x240 for the old tracer.
No deployment.json required. Tracer and 0.5 s line-loss hold kept from old host.
IMPORTANT: the ONNX file has no feature semantics or preprocessing metadata.
The SETTINGS below are explicit PROVISIONAL deployment assumptions, not verified
training settings. This camera tracer replaces the unavailable original perception.
Exact sim-to-real equivalence needs the original observation/export/perception code.
Raw network outputs are assumed LEFT/RIGHT DUTY, clipped only. No PID/boost.
"""
import argparse
from dataclasses import dataclass
import math
import struct
import time
import zlib
import numpy as np
# ---------- PROVISIONAL settings: match your original training/export ----------
# Physical values: e_y in normalized image coordinates; e_theta_rad in radians;
# line_confidence in [0,1]; output-shaft rpm; previous applied motor duties.
FEATURE_ORDER = ('e_y', 'e_theta_rad', 'line_confidence', 'rpm_left',
                 'rpm_right', 'duty_prev_left', 'duty_prev_right')
FEATURE_OFFSET = np.array([0,0,0,0,0,0,0], dtype=np.float32)
FEATURE_SCALE = np.array([1,1,1,1/420.0,1/420.0,1,1], dtype=np.float32)
# Network normalization: (physical ordered features - offset) * scale.
# If training used raw RPM, change the two 1/420 values to 1.
OUTPUT_ORDER = ('left','right')
MAX_DUTY = 0.75
MAX_LINE_LOSS_S = 0.5
EY_SIGN = 1.0
ANGLE_SIGN = 1.0
BLACK_LINE = True
ROI_BOTTOM = 0.90
ROI_TOP = 0.40
LOOKAHEAD_ROW = 0.60
MIN_TRACE_ROWS = 5
MAX_TRACE_JUMP_PX = 45.0
MIN_LINE_WIDTH_PX = 2
MAX_LINE_WIDTH_PX = 180
MIN_IMAGE_CONTRAST = 130.0
# -----------------------------------------------------------------------------
MAGIC = b'LFP1'
HEADER = struct.Struct('<4sIIHHffffII')
MAX_PACKET = HEADER.size + 320*240
@dataclass
class Frame:
    seq: int
    timestamp_us: int
    image: np.ndarray
    rpm_left: float
    rpm_right: float
    duty_prev_left: float
    duty_prev_right: float
    received_at: float
class FrameParser:
    def __init__(self):
        self.buffer = bytearray()
        self.bytes_received = 0
        self.bad_crc = 0
        self.frame_count = 0
        self.messages = []

    def reset_buffer(self):
        self.buffer.clear()
        self.messages.clear()
    def feed(self, data):
        self.bytes_received += len(data)
        self.buffer.extend(data)
        frames=[]
        while True:
            # Read diagnostic text only outside a validated binary packet.
            if self.buffer.startswith(b'!LF'):
                end=self.buffer.find(b'\n')
                if end<0:
                    if len(self.buffer)>4096:
                        del self.buffer[0]
                        continue
                    break
                self.messages.append(bytes(self.buffer[:end]).decode('utf-8','replace'))
                del self.buffer[:end+1]
                continue
            start=self.buffer.find(MAGIC)
            status=self.buffer.find(b'!LF')
            if status>=0 and (start<0 or status<start):
                del self.buffer[:status]
                continue
            if start<0:
                self.buffer=self.buffer[-3:]
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer)<HEADER.size:
                break
            fields=HEADER.unpack_from(self.buffer)
            _,seq,stamp,w,h,rl,rr,dl,dr,size,crc=fields
            if (w,h) not in ((160,120),(320,240)) or size!=w*h or not all(map(math.isfinite,(rl,rr,dl,dr))):
                del self.buffer[0]
                continue
            total=HEADER.size+size
            if len(self.buffer)<total:
                break
            payload=bytes(self.buffer[HEADER.size:total])
            if zlib.crc32(payload)&0xffffffff != crc:
                # Re-synchronize even after an incomplete/aborted device write.
                self.bad_crc += 1
                del self.buffer[0]
                continue
            del self.buffer[:total]
            image=np.frombuffer(payload,dtype=np.uint8).reshape(h,w)
            if (w,h)==(160,120):
                # Preserve old tracer's pixel coordinates/width/jump thresholds.
                # Upsampling does not restore detail lost by lower capture resolution.
                image=np.repeat(np.repeat(image,2,axis=0),2,axis=1)
            frames.append(Frame(seq,stamp,image,rl,rr,dl,dr,time.monotonic()))
            self.frame_count += 1
        if len(self.buffer)>2*MAX_PACKET:
            self.buffer.clear()
        return frames
class Policy:
    def __init__(self, path):
        import onnxruntime as ort
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=2
        opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(path,opts,providers=['CPUExecutionProvider'])
        inputs,outputs=self.session.get_inputs(),self.session.get_outputs()
        if len(inputs)!=1 or inputs[0].shape!=[1,7] or inputs[0].type!='tensor(float)':
            raise ValueError('Expected float32 input [1,7]')
        if len(outputs)!=1 or outputs[0].shape!=[1,2]:
            raise ValueError('Expected output [1,2]')
        self.input_name,self.output_name=inputs[0].name,outputs[0].name
    def infer(self, observation):
        x=np.asarray(observation,dtype=np.float32).reshape(1,7)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite observation')
        y=self.session.run([self.output_name],{self.input_name:x})[0].reshape(2)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite action')
        return tuple(map(float,y))
def otsu_threshold(image):
    hist=np.bincount(image.ravel(),minlength=256).astype(np.float64)
    weight=np.cumsum(hist)
    moment=np.cumsum(hist*np.arange(256))
    total=weight[-1];total_moment=moment[-1]
    denominator=weight*(total-weight)
    score=np.zeros(256)
    np.divide((total_moment*weight-moment*total)**2,denominator,
              out=score,where=denominator>0)
    return int(np.argmax(score))
class LineTracer:
    """Simple geometric black/white-line tracer; not the original simulator tracer."""
    def __init__(self):
        self.bottom_center=None
    def detect(self, image):
        h,w=image.shape
        roi=image[int(h*ROI_TOP):int(h*ROI_BOTTOM)+1]
        if int(roi.max())-int(roi.min())<MIN_IMAGE_CONTRAST:
            return None,[]
        threshold=otsu_threshold(roi)
        rows=np.linspace(int(h*ROI_BOTTOM),int(h*ROI_TOP),12).astype(int)
        target=(w-1)/2 if self.bottom_center is None else self.bottom_center
        points=[]
        for row in rows:
            # A 3-row median reduces isolated noise without OpenCV.
            stripe=np.median(image[max(0,row-1):min(h,row+2)],axis=0)
            mask=stripe<=threshold if BLACK_LINE else stripe>threshold
            padded=np.pad(mask.astype(np.int8),(1,1))
            diff=np.diff(padded)
            starts=np.flatnonzero(diff==1);ends=np.flatnonzero(diff==-1)
            candidates=[(start+end-1)/2 for start,end in zip(starts,ends)
                        if MIN_LINE_WIDTH_PX<=end-start<=MAX_LINE_WIDTH_PX]
            if not candidates:
                continue
            center=min(candidates,key=lambda x:abs(x-target))
            if abs(center-target)>MAX_TRACE_JUMP_PX:
                continue
            points.append((center,row));target=center
        if len(points)<MIN_TRACE_ROWS:
            return None,points
        xs=np.array([p[0] for p in points]);ys=np.array([p[1] for p in points])
        slope,intercept=np.polyfit(ys,xs,1)
        residual=np.abs(xs-(slope*ys+intercept))
        keep=residual<12
        if np.count_nonzero(keep)<MIN_TRACE_ROWS:
            return None,points
        slope,intercept=np.polyfit(ys[keep],xs[keep],1)
        center=slope*(LOOKAHEAD_ROW*h)+intercept
        ey=EY_SIGN*np.clip((center-(w-1)/2)/(w/2),-1,1)
        # Image y points downward: minus dx/dy is heading toward image top.
        angle=ANGLE_SIGN*math.atan(-float(slope))
        self.bottom_center=float(xs[0])
        return (float(ey),float(angle)),points
class ObservationBuilder:
    def __init__(self):
        self.tracer=LineTracer()
        self.last_line=None
        self.last_valid=None
    def build(self, frame):
        line,points=self.tracer.detect(frame.image)
        now=frame.received_at
        if line is not None:
            self.last_line=line;self.last_valid=now
        loss=math.inf if self.last_valid is None else max(0,now-self.last_valid)
        can_drive=self.last_line is not None and loss<=MAX_LINE_LOSS_S
        if not can_drive:
            self.last_line=None;self.tracer.bottom_center=None
        ey,angle=self.last_line if self.last_line is not None else (0.,0.)
        confidence=max(0,1-loss/MAX_LINE_LOSS_S) if math.isfinite(loss) else 0.
        values=dict(e_y=ey,e_theta_rad=angle,line_confidence=confidence,
                    rpm_left=frame.rpm_left,rpm_right=frame.rpm_right,
                    duty_prev_left=frame.duty_prev_left,duty_prev_right=frame.duty_prev_right)
        x=np.array([values[name] for name in FEATURE_ORDER],dtype=np.float32)
        x=(x-FEATURE_OFFSET)*FEATURE_SCALE
        return x,can_drive,values,points,loss
    def reset(self):
        self.last_line=None;self.last_valid=None;self.tracer.bottom_center=None


def send_command(port, command):
    data=command.encode('ascii')
    if port.write(data)!=len(data):
        raise IOError('Incomplete serial command write')


def restart_stream(port, frames, builder):
    # S now stops BOTH motors and streaming. G is required to resume.
    send_command(port,'S\n')
    time.sleep(.1)
    port.reset_input_buffer()
    frames.reset_buffer()
    builder.reset()
    send_command(port,'G\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='policy.onnx')
    parser.add_argument('--port',default='/dev/ttyACM0')
    parser.add_argument('--drive',action='store_true',help='Send policy duties; default sends zero duty')
    parser.add_argument('--camera-only',action='store_true',help='Camera/USB test without ONNX')
    parser.add_argument('--preview',action='store_true',help='Requires desktop display and OpenCV; Q stops')
    parser.add_argument('--check',action='store_true',help='ONNX synthetic inference only; no serial')
    args=parser.parse_args()
    if args.camera_only and (args.drive or args.check):
        parser.error('--camera-only cannot be combined with --drive or --check')

    policy=None if args.camera_only else Policy(args.model)
    if args.check:
        print('ONNX input: float32 [1,7]; output: [1,2]')
        print('Raw zero-input output:',policy.infer([0]*7))
        print('Inference OK; observation semantics are still unverified.')
        return

    import serial
    cv2=None
    if args.preview:
        import cv2
    print('DRIVE mode' if args.drive else 'MONITOR mode: motor duties remain zero',flush=True)
    print('Provisional observation order/scaling, perception and action mapping; see SETTINGS.',flush=True)
    frames=FrameParser()
    builder=ObservationBuilder()
    port=None
    try:
        # Exclusive access is Linux-only, appropriate for Raspberry Pi.
        port=serial.Serial(args.port,921600,timeout=.02,write_timeout=.2,exclusive=True)
        port.dtr=True
        port.rts=False
        time.sleep(.5)
        restart_stream(port,frames,builder)
        print(f'USB opened: {args.port}; G sent; waiting for 160x120 or 320x240 grayscale frames.',flush=True)
        last_frame=time.monotonic()
        last_log=0.
        last_seq=None
        fps_start=last_frame
        fps_frames=0
        while True:
            data=port.read(min(max(port.in_waiting,1),65536))
            received=frames.feed(data)
            for message in frames.messages:
                print('ESP32:',message,flush=True)
            frames.messages.clear()
            if received:
                frame=received[-1]  # Infer only on newest complete frame.
                if frame.seq==last_seq:
                    continue
                if last_seq is not None and frame.seq<last_seq:
                    builder.reset()  # MCU reboot/sequence wrap.
                last_seq=frame.seq
                last_frame=frame.received_at
                fps_frames+=len(received)
                x,valid,values,points,loss=builder.build(frame)
                proposed_left=proposed_right=0.
                if policy is not None and valid:
                    raw=policy.infer(x)
                    mapped=dict(zip(OUTPUT_ORDER,raw))
                    proposed_left=float(np.clip(mapped['left'],-MAX_DUTY,MAX_DUTY))
                    proposed_right=float(np.clip(mapped['right'],-MAX_DUTY,MAX_DUTY))
                processing_ms=(time.monotonic()-frame.received_at)*1000
                # This bounds host processing only. MCU separately bounds total reply age.
                fresh=processing_ms<120
                left=proposed_left if args.drive and valid and fresh else 0.
                right=proposed_right if args.drive and valid and fresh else 0.
                send_command(port,f'D {frame.seq} {left:.7f} {right:.7f}\n')
                now=time.monotonic()
                if now-last_log>=1:
                    fps=fps_frames/max(now-fps_start,.001)
                    print(
                        f'frame={frame.seq} fps={fps:.1f} line={int(valid)} '
                        f'conf={values["line_confidence"]:.2f} '
                        f'ey={values["e_y"]:+.3f} angle={values["e_theta_rad"]:+.3f} '
                        f'rpm=({frame.rpm_left:+.1f},{frame.rpm_right:+.1f}) '
                        f'policy=({proposed_left:+.3f},{proposed_right:+.3f}) '
                        f'sent=({left:+.3f},{right:+.3f}) '
                        f'applied_prev=({frame.duty_prev_left:+.3f},{frame.duty_prev_right:+.3f}) '
                        f'process={processing_ms:.1f}ms CRC_errors={frames.bad_crc}',
                        flush=True
                    )
                    last_log=now;fps_start=now;fps_frames=0
                if cv2 is not None:
                    image=cv2.cvtColor(frame.image,cv2.COLOR_GRAY2BGR)
                    for xpoint,ypoint in points:
                        cv2.circle(image,(round(xpoint),int(ypoint)),3,(0,255,0),-1)
                    cv2.putText(image,f'line {int(valid)} duty {left:+.2f} {right:+.2f}',
                                (5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
                    cv2.imshow('ESP32 camera / Pi ONNX',image)
            if cv2 is not None and cv2.waitKey(1)&0xff in (ord('q'),ord('Q')):
                break
            if time.monotonic()-last_frame>3.:
                print(
                    f'No valid frame for 3 seconds: bytes={frames.bytes_received}, '
                    f'frames={frames.frame_count}, CRC_errors={frames.bad_crc}. '
                    'Sending S then G to restart streaming.',flush=True
                )
                restart_stream(port,frames,builder)
                last_seq=None
                last_frame=time.monotonic()
                fps_start=last_frame;fps_frames=0
    except KeyboardInterrupt:
        print('\nStopped.',flush=True)
    except serial.SerialException as exc:
        print(f'USB/serial error: {exc}',flush=True)
        print('After an ESP32 reset, wait for /dev/ttyACM* to return and rerun.',flush=True)
        raise SystemExit(1)
    finally:
        if port is not None:
            try:
                send_command(port,'S\n')
            except (serial.SerialException,IOError):
                pass
            port.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__=='__main__':
    main()
