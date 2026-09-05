"""Development-only VAD dashboard. Run with ``python -m ghostpilot.system1.vad_debug``."""

import argparse
import asyncio
from dataclasses import asdict, replace
import io
import logging
import time
from typing import TYPE_CHECKING, Any
import wave

from .adapters.sounddevice_input import SoundDeviceAudioInput
from .audio import AudioFrame
from .config import AudioConfig, System1Config, VADConfig
from .runtime import System1Runtime, default_provider_registry
from .vad import EnergyVoiceActivityDetector

if TYPE_CHECKING:
    from fastapi import FastAPI


class MicrophoneTestRecorder:
    """Bounded, explicit debug capture of canonical frames for local playback."""

    def __init__(self, *, sample_rate: int = 16_000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.active = False
        self._pcm = bytearray()
        self._maximum_bytes = 0

    def start(self, duration_seconds: float) -> None:
        if self.active:
            raise RuntimeError("A microphone test is already recording")
        if not 1 <= duration_seconds <= 10:
            raise ValueError("Microphone test duration must be between 1 and 10 seconds")
        self._pcm.clear()
        self._maximum_bytes = round(
            self.sample_rate * self.channels * 2 * duration_seconds
        )
        self.active = True

    def observe(self, frame: AudioFrame) -> None:
        if not self.active or len(self._pcm) >= self._maximum_bytes:
            return
        remaining = self._maximum_bytes - len(self._pcm)
        self._pcm.extend(frame.data[:remaining])

    def finish_wav(self) -> tuple[bytes, float]:
        self.active = False
        pcm = bytes(self._pcm)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)
        duration = len(pcm) / (self.sample_rate * self.channels * 2)
        return output.getvalue(), duration


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GhostPilot System 1 — M3B Realtime Debug</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#070b12;color:#eaf0fb;--panel:#101722;--line:#243247;--muted:#8fa1ba;--cyan:#45d4ff;--green:#55e69b;--amber:#ffcc66;--red:#ff6b7a}
*{box-sizing:border-box}body{margin:0;padding:1.25rem;min-width:320px}main{max-width:1440px;margin:auto}header{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:1rem}h1{font-size:1.35rem;margin:0;letter-spacing:-.02em}.eyebrow,.label{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.12em}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:.8rem}.card{background:linear-gradient(145deg,#111a27,#0d141e);border:1px solid var(--line);border-radius:10px;padding:1rem;min-width:0}.span3{grid-column:span 3}.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}.hero-state{font:700 1.75rem ui-monospace,monospace;color:var(--cyan);margin-top:.65rem}.badge{display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .5rem;border:1px solid #344861;border-radius:999px;color:var(--muted);font:600 .75rem ui-monospace,monospace}.badge::before{content:'';width:.45rem;height:.45rem;border-radius:50%;background:currentColor}.ok{color:var(--green)}.warn{color:var(--amber)}.error{color:var(--red)}.live-text{font-size:1.25rem;line-height:1.5;min-height:3rem;margin:.65rem 0 0;overflow-wrap:anywhere}.final-text{color:#c7d4e7}.bar{height:12px;background:#1a2636;border-radius:2px;overflow:hidden;margin:.8rem 0 .4rem}.bar>div{height:100%;width:0;background:linear-gradient(90deg,#168db5,var(--green));transition:width .07s linear}dl{display:grid;grid-template-columns:minmax(8rem,max-content) 1fr;gap:.45rem .8rem;margin:.7rem 0 0;font-size:.88rem}dt{color:var(--muted)}dd{margin:0;font-family:ui-monospace,monospace;overflow-wrap:anywhere}select,input,button{width:100%;padding:.6rem .7rem;border-radius:6px;border:1px solid #32455f;background:#090f18;color:#eaf0fb;font:inherit}button{cursor:pointer;background:#162a40;font-weight:650}button:hover{border-color:var(--cyan)}button:disabled{cursor:not-allowed;opacity:.45}.controls{display:flex;gap:.45rem;margin-top:.65rem}.controls>*{flex:1}.status-line{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.7rem}.timeline,.trace{margin:.7rem 0 0;padding:0;list-style:none;max-height:340px;overflow:auto;font:.79rem/1.45 ui-monospace,monospace}.timeline li{display:grid;grid-template-columns:7.5rem 15rem 1fr;gap:.6rem;padding:.42rem 0;border-bottom:1px solid #1c293b}.trace li{padding:.35rem 0;border-bottom:1px solid #1c293b;color:#b9c9de}.timeline time{color:var(--muted)}.timeline strong{color:#b9c9de}.error-text{color:var(--red)}.muted{color:var(--muted);font-size:.82rem}.metric{font:650 1.1rem ui-monospace,monospace;margin-top:.4rem}audio{display:block;width:100%;height:2.5rem;margin-top:.6rem;accent-color:var(--cyan)}@media(max-width:980px){.span3,.span4,.span5,.span6,.span7,.span8{grid-column:span 6}}@media(max-width:650px){body{padding:.75rem}.grid>*{grid-column:span 12!important}header{align-items:flex-start;flex-direction:column}.timeline li{grid-template-columns:1fr}.controls{flex-wrap:wrap}.controls>*{flex:1 1 45%}}
</style></head><body><main><header><div><div class="eyebrow">Realtime observability console</div><h1>GhostPilot System 1 · M3B</h1></div><span id="dashboardSocket" class="badge warn">DASHBOARD CONNECTING</span></header><div class="grid">
<section class="card span4"><div class="label">System 1 State</div><div id="systemState" class="hero-state">LISTENING</div><div class="status-line"><span id="commitBadge" class="badge">NOT COMMITTED</span><span id="endpointBadge" class="badge">ENDPOINT IDLE</span></div></section>
<section class="card span4"><div class="label">Voice Activity</div><div class="status-line"><span id="vadBadge" class="badge">SILENT</span><span id="lastVad" class="badge">NO EVENT</span></div><div class="bar"><div id="level"></div></div><div><span id="levelText">0.000</span> RMS · threshold <span id="threshold">—</span></div><div class="muted">Probability <span id="probabilityText">—</span></div></section>
<section class="card span4"><div class="label">STT Service</div><div class="status-line"><span id="sttBadge" class="badge warn">DISCONNECTED</span><span id="healthBadge" class="badge">UNKNOWN</span></div><dl><dt>Provider</dt><dd id="provider">—</dd><dt>Model</dt><dd id="model">—</dd><dt>Device</dt><dd id="sttDevice">—</dd><dt>Lookahead</dt><dd id="lookahead">—</dd><dt>Stream latency</dt><dd id="streamLatency">—</dd><dt>Warmed</dt><dd id="warmed">—</dd><dt>Reconnects</dt><dd id="reconnects">0</dd><dt>Last error</dt><dd id="sttError">—</dd></dl><div class="controls"><button data-stt="reconnect">Reconnect</button><button data-stt="reset">Reset session</button><button data-stt="ping">Ping</button></div></section>
<section class="card span12"><div class="label">Current Partial</div><div id="partial" class="live-text">Waiting for speech…</div></section>
<section class="card span6"><div class="label">Current Final</div><div id="final" class="live-text final-text">—</div></section>
<section class="card span6"><div class="label">Best Transcript</div><div id="best" class="live-text final-text">—</div></section>
<section class="card span5"><div class="label">Turn + Segment</div><dl><dt>Turn ID</dt><dd id="turnId">—</dd><dt>Segment</dt><dd id="segment">—</dd><dt>Latest final</dt><dd id="latestFinal">false</dd><dt>Turn state</dt><dd id="turnState">LISTENING</dd><dt>Endpoint</dt><dd id="endpoint">IDLE</dd><dt>Timer</dt><dd id="endpointTimer">—</dd></dl></section>
<section class="card span7"><div class="label">Latency</div><div class="grid"><div class="span3"><div class="muted">Start → partial</div><div id="latPartial" class="metric">—</div></div><div class="span3"><div class="muted">Stop → final</div><div id="latFinal" class="metric">—</div></div><div class="span3"><div class="muted">Stop → commit</div><div id="latCommit" class="metric">—</div></div><div class="span3"><div class="muted">Response age</div><div id="responseAge" class="metric">—</div></div></div></section>
<section class="card span4"><div class="label">Microphone + Listen Back</div><select id="device" aria-label="Microphone input device"><option>Loading devices…</option></select><button id="useDevice" style="margin-top:.5rem">Use selected input</button><button id="recordMic" style="margin-top:.5rem" disabled>Record 4 seconds</button><div id="micTestStatus" class="muted" style="margin-top:.5rem">Select an input first</div><audio id="micPlayback" controls hidden preload="metadata"></audio><div id="inputStatus" class="muted" style="margin-top:.5rem">No input connected</div></section>
<section class="card span4"><div class="label">Audio Stream</div><dl><dt>Format</dt><dd id="format">—</dd><dt>Frames sent</dt><dd id="framesSent">0</dd><dt>Bytes sent</dt><dd id="bytesSent">0</dd><dt>STT queue</dt><dd id="sttQueue">0</dd><dt>STT drops</dt><dd id="sttDrops">0</dd><dt>Input drops</dt><dd id="inputDrops">0</dd></dl></section>
<section class="card span4"><div class="label">Buffers</div><dl><dt>Input queue</dt><dd id="inputQueue">0</dd><dt>Pre-roll</dt><dd id="preRoll">—</dd><dt>Turn audio</dt><dd id="buffer">—</dd></dl><div id="mockControls"><input id="mockTranscript" placeholder="Mock transcript only"><div class="controls"><button id="sendPartial">Partial</button><button id="sendFinal">Final</button></div><div id="injectStatus" class="muted">Available in mock mode</div></div></section>
<section class="card span7"><div class="label">STT Protocol Debug</div><div class="status-line"><span id="protocolVerdict" class="badge">WAITING FOR AUDIO</span></div><dl><dt>Connection</dt><dd id="debugConnection">—</dd><dt>Active binding</dt><dd id="debugBinding">—</dd><dt>Segment queued</dt><dd id="debugQueued">0 frames · 0 bytes</dd><dt>Segment sent</dt><dd id="debugSent">0 frames · 0 bytes</dd><dt>Last send age</dt><dd id="debugSendAge">—</dd><dt>Last control</dt><dd id="debugControl">—</dd><dt>Last response</dt><dd id="debugResponse">—</dd><dt>Responses</dt><dd id="debugResponses">0 total · 0 partial · 0 final</dd><dt>Ignored</dt><dd id="debugIgnored">0</dd><dt>Ignore reason</dt><dd id="debugIgnoreReason">—</dd></dl></section>
<section class="card span5"><div class="label">STT Protocol Trace</div><ul id="protocolTrace" class="trace"><li>Waiting for connection…</li></ul></section>
<section class="card span12"><div class="label">Event Timeline</div><ul id="events" class="timeline"><li><time>—</time><strong>Waiting for events</strong><span></span></li></ul></section>
</div></main><script>
const $=id=>document.getElementById(id); const ms=v=>v==null?'—':`${Number(v).toFixed(1)} ms`; const value=v=>(v===null||v===undefined||v==='')?'—':v;
function setBadge(id,text,kind=''){const el=$(id);el.textContent=text;el.className=`badge ${kind}`}
function renderTrace(items){const list=$('protocolTrace');list.replaceChildren();for(const message of [...(items||[])].reverse()){const li=document.createElement('li');li.textContent=message;list.append(li)}if(!list.children.length){const li=document.createElement('li');li.textContent='No protocol activity yet';list.append(li)}}
function snapshot(s){const stt=s.stt||{},lat=s.latency||{};$('systemState').textContent=s.turn_state;$('turnState').textContent=s.turn_state;setBadge('commitBadge',s.turn_committed?'COMMITTED':'NOT COMMITTED',s.turn_committed?'ok':'');setBadge('endpointBadge',`ENDPOINT ${s.endpoint_state}`,s.endpoint_pending?'warn':s.endpoint_state==='COMMITTED'?'ok':'');const speaking=s.vad_state==='SPEAKING';setBadge('vadBadge',speaking?'SPEAKING':'SILENT',speaking?'ok':'');$('level').style.width=`${Math.min(100,(s.audio_level||0)*500)}%`;$('levelText').textContent=Number(s.audio_level||0).toFixed(3);$('probabilityText').textContent=s.vad_probability==null?'—':Number(s.vad_probability).toFixed(2);$('threshold').textContent=value(s.vad_threshold);setBadge('sttBadge',stt.connected?(stt.ready===false?'CONNECTED / WAITING':'CONNECTED'):'DISCONNECTED',stt.connected?'ok':'error');setBadge('healthBadge',String(stt.health_status||'unknown').toUpperCase(),stt.health_status==='healthy'||stt.health_status==='mock'?'ok':stt.health_status==='unhealthy'?'error':'');$('provider').textContent=value(stt.provider);$('model').textContent=value(stt.model);$('sttDevice').textContent=value(stt.device);$('lookahead').textContent=value(stt.lookahead);$('streamLatency').textContent=ms(stt.streaming_latency_ms);$('warmed').textContent=String(Boolean(stt.warmed_up));$('reconnects').textContent=value(stt.reconnect_count);$('sttError').textContent=value(stt.last_error);$('sttError').className=stt.last_error?'error-text':'';$('partial').textContent=s.partial_transcript||'Waiting for speech…';$('final').textContent=s.final_transcript||'—';$('best').textContent=s.best_transcript||'—';$('turnId').textContent=s.current_turn_id||'—';$('segment').textContent=s.current_turn_id?s.speech_segment_id:'—';$('latestFinal').textContent=String(Boolean(s.latest_segment_final));$('endpoint').textContent=s.endpoint_state;$('endpointTimer').textContent=ms(s.endpoint_remaining_ms);$('latPartial').textContent=ms(lat.speech_start_to_first_partial_ms);$('latFinal').textContent=ms(lat.speech_stop_to_segment_final_ms);$('latCommit').textContent=ms(lat.speech_stop_to_commit_ms);$('responseAge').textContent=ms(lat.latest_stt_response_age_ms);$('format').textContent=`${s.sample_rate} Hz · ${s.frame_duration_ms} ms · PCM16 mono`;$('framesSent').textContent=value(stt.frames_sent);$('bytesSent').textContent=Number(stt.audio_bytes_sent||0).toLocaleString();$('sttQueue').textContent=`${value(stt.send_queue_depth)} / ${value(stt.send_queue_capacity)}`;$('sttDrops').textContent=value(stt.stt_dropped_frames);$('inputDrops').textContent=s.frames_dropped;$('inputQueue').textContent=s.audio_queue_size;$('preRoll').textContent=`${s.pre_roll_audio_seconds}s · ${s.pre_roll_frame_count} frames`;$('buffer').textContent=`${s.buffered_audio_seconds}s`;$('mockControls').style.display=stt.provider==='mock'?'block':'none';$('inputStatus').textContent=s.audio_connected?`Connected · device ${s.audio_device}`:'No input connected';$('recordMic').disabled=!s.audio_connected;$('debugConnection').textContent=`generation ${value(stt.connection_generation)} · ${stt.connected?'connected':'disconnected'}`;$('debugBinding').textContent=stt.active_turn_id?`${stt.active_turn_id} / segment ${stt.active_segment_id}`:'—';$('debugQueued').textContent=`${value(stt.segment_frames_queued)} frames · ${Number(stt.segment_bytes_queued||0).toLocaleString()} bytes`;$('debugSent').textContent=`${value(stt.segment_frames_sent)} frames · ${Number(stt.segment_bytes_sent||0).toLocaleString()} bytes`;$('debugSendAge').textContent=ms(stt.latest_audio_send_age_ms);$('debugControl').textContent=value(stt.last_control_sent);$('debugResponse').textContent=value(stt.last_response_type);$('debugResponses').textContent=`${value(stt.messages_received)} total · ${value(stt.partial_responses)} partial · ${value(stt.final_responses)} final`;$('debugIgnored').textContent=value(stt.ignored_responses);$('debugIgnoreReason').textContent=value(stt.last_ignored_reason);const sent=Number(stt.segment_frames_sent||0),responses=Number(stt.segment_partial_responses||0)+Number(stt.segment_final_responses||0);if(!stt.connected)setBadge('protocolVerdict','WEBSOCKET DISCONNECTED','error');else if(sent===0)setBadge('protocolVerdict','WAITING FOR SPEECH AUDIO','');else if(responses===0)setBadge('protocolVerdict','PCM SENT · NO TRANSCRIPT RESPONSE','warn');else if(Number(stt.segment_final_responses||0)>0)setBadge('protocolVerdict','SEGMENT FINAL RECEIVED','ok');else setBadge('protocolVerdict','PARTIAL RECEIVED','ok');renderTrace(stt.trace)}
function summary(e){const parts=[];if(e.turn_id)parts.push(e.turn_id);if(e.segment_id!=null)parts.push(`segment ${e.segment_id}`);if(e.text)parts.push(`“${e.text}”`);if(e.transcript)parts.push(`“${e.transcript}”`);if(e.status)parts.push(e.status);if(e.detail)parts.push(e.detail);return parts.join(' · ')}
function event(item){const e=item.event,list=$('events');if(list.dataset.empty!=='false'){list.replaceChildren();list.dataset.empty='false'}const li=document.createElement('li'),time=document.createElement('time'),name=document.createElement('strong'),payload=document.createElement('span');time.textContent=new Date(item.timestamp*1000).toLocaleTimeString([], {hour12:false,fractionalSecondDigits:3});name.textContent=e.name;payload.textContent=summary(e);li.append(time,name,payload);list.prepend(li);while(list.children.length>100)list.lastChild.remove();if(e.name==='audio.speech_started'||e.name==='audio.speech_stopped')setBadge('lastVad',e.name.replace('audio.','').toUpperCase(),e.name.endsWith('started')?'ok':'warn')}
async function loadDevices(){const response=await fetch('/api/audio-devices'),data=await response.json(),select=$('device');select.replaceChildren();for(const device of data.devices){const option=document.createElement('option');option.value=device.id;option.textContent=`${device.id} — ${device.name} (${device.host_api})`;if(String(data.selected_device)===String(device.id))option.selected=true;select.append(option)}}
const socket=new WebSocket(`ws://${location.host}/ws`);socket.onopen=()=>{setBadge('dashboardSocket','DASHBOARD CONNECTED','ok');loadDevices()};socket.onclose=()=>setBadge('dashboardSocket','DASHBOARD DISCONNECTED','error');socket.onmessage=message=>{const data=JSON.parse(message.data);if(data.type==='snapshot')snapshot(data.snapshot);if(data.type==='event')event(data);if(data.type==='audio_selected')$('inputStatus').textContent=`Connected · device ${data.device}`;if(data.type==='audio_error'||data.type==='stt_control_error')$('sttError').textContent=data.detail};$('useDevice').onclick=()=>socket.send(JSON.stringify({type:'select_audio_device',device:Number($('device').value)}));document.querySelectorAll('[data-stt]').forEach(button=>button.onclick=()=>socket.send(JSON.stringify({type:'stt_control',action:button.dataset.stt})));
async function injectTranscript(isFinal){const text=$('mockTranscript').value.trim();if(!text)return;const response=await fetch('/api/mock-transcript',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,is_final:isFinal})}),data=await response.json();$('injectStatus').textContent=response.ok?`Sent ${isFinal?'final':'partial'} · ${data.turn_id}`:(data.detail||'Injection failed')}$('sendPartial').onclick=()=>injectTranscript(false);$('sendFinal').onclick=()=>injectTranscript(true);
let microphoneRecordingUrl=null;async function recordMicrophone(){const button=$('recordMic'),status=$('micTestStatus'),audio=$('micPlayback');button.disabled=true;status.textContent='Recording… speak normally for 4 seconds';audio.hidden=true;try{const response=await fetch('/api/microphone-test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({duration_seconds:4})});if(!response.ok){const data=await response.json();throw new Error(data.detail||'Microphone test failed')}const blob=await response.blob();if(microphoneRecordingUrl)URL.revokeObjectURL(microphoneRecordingUrl);microphoneRecordingUrl=URL.createObjectURL(blob);audio.src=microphoneRecordingUrl;audio.hidden=false;const seconds=Number(response.headers.get('X-Recorded-Duration')||0);status.textContent=seconds>0?`Captured ${seconds.toFixed(2)} seconds · press play to listen`:'No audio frames were captured'}catch(error){status.textContent=error.message}finally{button.disabled=false}}$('recordMic').onclick=recordMicrophone;
</script></body></html>"""


def create_app(runtime: System1Runtime) -> "FastAPI":
    """Build a debug-only app without making FastAPI a domain dependency."""
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, Response
    except ImportError as error:
        raise RuntimeError("Install debug support: pip install -e '.[vad-debug]'") from error

    app = FastAPI(title="GhostPilot VAD Debug", docs_url=None, redoc_url=None)
    microphone_recorder = MicrophoneTestRecorder(
        sample_rate=runtime.config.audio.sample_rate,
        channels=runtime.config.audio.channels,
    )
    runtime.set_audio_frame_observer(microphone_recorder.observe)

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE

    @app.get("/api/audio-devices")
    async def audio_devices() -> dict[str, object]:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("Install audio support: pip install -e '.[audio]'") from error
        host_apis = sd.query_hostapis()
        devices = [
            {
                "id": index,
                "name": device["name"],
                "host_api": host_apis[device["hostapi"]]["name"],
            }
            for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0
        ]
        return {"devices": devices, "selected_device": runtime.config.audio.device}

    @app.post("/api/microphone-test")
    async def microphone_test(payload: dict[str, Any]) -> "Response":
        if not runtime.audio_connected:
            raise HTTPException(status_code=409, detail="Select and connect a microphone first")
        try:
            duration = float(payload.get("duration_seconds", 4))
            microphone_recorder.start(duration)
        except (TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        try:
            await asyncio.sleep(duration)
        finally:
            wav_bytes, recorded_duration = microphone_recorder.finish_wav()
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-Recorded-Duration": f"{recorded_duration:.3f}",
                "Content-Disposition": 'inline; filename="ghostpilot-microphone-test.wav"',
            },
        )

    @app.post("/api/mock-transcript")
    async def inject_mock_transcript(payload: dict[str, Any]) -> dict[str, object]:
        emit = getattr(runtime.stt, "emit", None)
        text = payload.get("text")
        if not callable(emit):
            raise HTTPException(status_code=409, detail="The active STT provider does not support injection")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text must be a non-empty string")
        turn_id = payload.get("turn_id") or runtime.state.current_turn
        if not isinstance(turn_id, str):
            raise HTTPException(status_code=409, detail="Start a user turn before injecting a transcript")
        await emit(
            text.strip(),
            is_final=bool(payload.get("is_final")),
            turn_id=turn_id,
            segment_id=runtime.transcripts.segment_id,
        )
        return {
            "turn_id": turn_id,
            "segment_id": runtime.transcripts.segment_id,
            "is_final": bool(payload.get("is_final")),
        }

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        events = runtime.events.subscribe(maxsize=256)
        event_task = asyncio.create_task(events.get())
        command_task = asyncio.create_task(websocket.receive_json())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {event_task, command_task}, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    await websocket.send_json({"type": "snapshot", "snapshot": runtime.debug_snapshot()})
                if event_task in done:
                    event = event_task.result()
                    event_payload = asdict(event)
                    if runtime.state.current_turn and "turn_id" not in event_payload:
                        event_payload["turn_id"] = runtime.state.current_turn
                    if "segment_id" not in event_payload:
                        event_payload["segment_id"] = runtime.transcripts.segment_id
                    await websocket.send_json(
                        {"type": "event", "timestamp": time.time(), "event": event_payload}
                    )
                    await websocket.send_json({"type": "snapshot", "snapshot": runtime.debug_snapshot()})
                    event_task = asyncio.create_task(events.get())
                if command_task in done:
                    command = command_task.result()
                    if command.get("type") == "select_audio_device":
                        try:
                            device = int(command["device"])
                            config = replace(runtime.config, audio=replace(runtime.config.audio, device=device))
                            await runtime.configure_audio_input(
                                SoundDeviceAudioInput(config.audio),
                                EnergyVoiceActivityDetector(config.vad),
                                config=config,
                            )
                        except (KeyError, TypeError, ValueError, RuntimeError) as error:
                            await websocket.send_json({"type": "audio_error", "detail": str(error)})
                        else:
                            await websocket.send_json({"type": "audio_selected", "device": device})
                    elif command.get("type") == "stt_control":
                        action = command.get("action")
                        method = getattr(runtime.stt, str(action), None)
                        if action not in {"reconnect", "reset", "ping"} or not callable(method):
                            await websocket.send_json(
                                {"type": "stt_control_error", "detail": "Unsupported STT control"}
                            )
                        else:
                            try:
                                await method()
                            except Exception as error:
                                await websocket.send_json(
                                    {"type": "stt_control_error", "detail": str(error)}
                                )
                    command_task = asyncio.create_task(websocket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            event_task.cancel()
            command_task.cancel()
            runtime.events.unsubscribe(events)

    return app


async def run(
    host: str,
    port: int,
    device: str | None,
    threshold: float,
    debug_stt: bool = False,
) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install debug support: pip install -e '.[vad-debug]'") from error

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if debug_stt:
        logging.getLogger("ghostpilot.system1.adapters.nemotron_stt").setLevel(logging.DEBUG)
    audio_device: int | str | None = int(device) if device and device.isdigit() else device
    base_config = System1Config.from_env()
    config = replace(
        base_config,
        audio=AudioConfig(device=audio_device),
        vad=VADConfig(speech_threshold=threshold),
    )
    runtime = System1Runtime.from_config(config, default_provider_registry(config))
    await runtime.start()
    server = uvicorn.Server(uvicorn.Config(create_app(runtime), host=host, port=port, log_level="info"))
    try:
        await server.serve()
    finally:
        await runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GhostPilot's local microphone/VAD debug UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="sounddevice input device name or index")
    parser.add_argument("--list-devices", action="store_true", help="print available input devices and exit")
    parser.add_argument("--threshold", type=float, default=0.015)
    parser.add_argument(
        "--debug-stt",
        action="store_true",
        help="log aggregate PCM sends, controls, and every STT response",
    )
    args = parser.parse_args()
    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("Install audio support: pip install -e '.[audio]'") from error
        print(sd.query_devices())
        return
    asyncio.run(run(args.host, args.port, args.device, args.threshold, args.debug_stt))


if __name__ == "__main__":
    main()
