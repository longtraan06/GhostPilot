"""Development-only VAD dashboard. Run with ``python -m ghostpilot.system1.vad_debug``."""

import argparse
import asyncio
from dataclasses import asdict, replace
import time
from typing import TYPE_CHECKING, Any

from .adapters.sounddevice_input import SoundDeviceAudioInput
from .config import AudioConfig, System1Config, VADConfig
from .runtime import System1Runtime
from .vad import EnergyVoiceActivityDetector

if TYPE_CHECKING:
    from fastapi import FastAPI


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GhostPilot VAD Debug</title><style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; background:#10151e; color:#e8eef9; }
body { max-width:760px; margin:0 auto; padding:2rem; } h1 { margin:0 0 1.5rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:1rem; }
.card { background:#192231; border:1px solid #2d405b; border-radius:12px; padding:1rem; }
.label { color:#9fb0c9; font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }
#state { font-size:2rem; font-weight:700; } .listening { color:#75b7ff; } .speaking { color:#62e6a7; }
.bar { height:14px; background:#26364e; border-radius:7px; overflow:hidden; margin-top:.5rem; }
.bar > div { height:100%; width:0; background:#62e6a7; transition:width .08s linear; }
dl { display:grid; grid-template-columns:max-content 1fr; gap:.45rem .8rem; margin:0; } dt { color:#9fb0c9; } dd { margin:0; }
#events { list-style:none; padding:0; max-height:280px; overflow:auto; } #events li { border-bottom:1px solid #2d405b; padding:.45rem 0; color:#cbd7e9; }
select, input, button { width:100%; margin-top:.55rem; padding:.55rem; border-radius:7px; border:1px solid #405878; background:#10151e; color:#e8eef9; font:inherit; } button { background:#2570c9; border:0; cursor:pointer; } #inputStatus, #injectStatus { display:block; margin-top:.6rem; color:#9fb0c9; } .buttons { display:flex; gap:.5rem; } .buttons button { flex:1; }
</style></head><body><h1>GhostPilot VAD Debug</h1><div class="grid">
<section class="card"><div class="label">Input device</div><select id="device" aria-label="Microphone input device"><option>Loading devices…</option></select><button id="useDevice">Use selected input</button><span id="inputStatus">No input connected</span></section>
<section class="card"><div class="label">VAD state</div><div id="state" class="listening">● LISTENING</div></section>
<section class="card"><div class="label">Audio level</div><div class="bar"><div id="level"></div></div><span id="levelText">0.00</span></section>
<section class="card"><div class="label">VAD probability</div><div class="bar"><div id="probability"></div></div><span id="probabilityText">—</span></section>
<section class="card"><div class="label">System 1 state</div><strong id="turn">LISTENING</strong></section>
<section class="card"><div class="label">Transcript and endpoint</div><dl><dt>Turn</dt><dd id="turnId">—</dd><dt>Partial</dt><dd id="partial">—</dd><dt>Final</dt><dd id="final">—</dd><dt>Endpoint</dt><dd id="endpoint">IDLE</dd></dl></section>
<section class="card"><div class="label">Inject mock transcript</div><input id="mockTranscript" placeholder="Transcript text"><div class="buttons"><button id="sendPartial">Send partial</button><button id="sendFinal">Send final</button></div><span id="injectStatus">Mock STT only</span></section>
<section class="card"><div class="label">Debug metrics</div><dl>
<dt>Format</dt><dd id="format">—</dd><dt>Input queue</dt><dd id="queue">—</dd><dt>Dropped frames</dt><dd id="dropped">—</dd><dt>Pre-roll buffer</dt><dd id="preRoll">—</dd><dt>Turn buffer</dt><dd id="buffer">—</dd>
</dl></section><section class="card"><div class="label">Recent events</div><ul id="events"></ul></section></div>
<script>
const byId = id => document.getElementById(id);
function bar(id, value) { byId(id).style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`; }
function snapshot(s) { const speaking = s.vad_state === 'SPEAKING'; const state=byId('state'); state.textContent=`● ${s.vad_state}`; state.className=speaking?'speaking':'listening';
 bar('level', s.audio_level); byId('levelText').textContent=Number(s.audio_level).toFixed(2); bar('probability', s.vad_probability); byId('probabilityText').textContent=s.vad_probability == null ? '—' : Number(s.vad_probability).toFixed(2);
 byId('turn').textContent=s.turn_state; byId('turnId').textContent=s.current_turn_id || '—'; byId('partial').textContent=s.partial_transcript || '—'; byId('final').textContent=s.final_transcript || '—'; byId('endpoint').textContent=s.endpoint_pending ? `${s.endpoint_state} (timer)` : s.endpoint_state; byId('format').textContent=`${s.sample_rate} Hz · ${s.frame_duration_ms} ms`; byId('queue').textContent=s.audio_queue_size; byId('dropped').textContent=s.frames_dropped; byId('preRoll').textContent=`${s.pre_roll_audio_seconds}s (${s.pre_roll_frame_count} frames)`; byId('buffer').textContent=`${s.buffered_audio_seconds}s`; }
function event(e) { const li=document.createElement('li'); li.textContent=`${new Date(e.timestamp * 1000).toLocaleTimeString()}  ${e.event.name}`; const list=byId('events'); list.prepend(li); while(list.children.length>40) list.lastChild.remove(); }
async function loadDevices() { const response=await fetch('/api/audio-devices'); const data=await response.json(); const select=byId('device'); select.replaceChildren(); for(const device of data.devices) { const option=document.createElement('option'); option.value=device.id; option.textContent=`${device.id} — ${device.name} (${device.host_api})`; if(String(data.selected_device)===String(device.id)) option.selected=true; select.append(option); } }
const socket = new WebSocket(`ws://${location.host}/ws`); socket.onopen=loadDevices; socket.onmessage = message => { const data=JSON.parse(message.data); if(data.type==='snapshot') { snapshot(data.snapshot); byId('inputStatus').textContent=data.snapshot.audio_connected ? `Connected: ${data.snapshot.audio_device}` : 'No input connected'; } if(data.type==='event') event(data); if(data.type==='audio_selected') byId('inputStatus').textContent=`Connected: ${data.device}`; if(data.type==='audio_error') byId('inputStatus').textContent=`Could not connect: ${data.detail}`; }; byId('useDevice').onclick=()=>socket.send(JSON.stringify({type:'select_audio_device', device:Number(byId('device').value)}));
async function injectTranscript(isFinal) { const text=byId('mockTranscript').value.trim(); if(!text) return; const response=await fetch('/api/mock-transcript',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,is_final:isFinal})}); const data=await response.json(); byId('injectStatus').textContent=response.ok ? `Sent ${isFinal ? 'final' : 'partial'} for ${data.turn_id}` : (data.detail || 'Injection failed'); }
byId('sendPartial').onclick=()=>injectTranscript(false); byId('sendFinal').onclick=()=>injectTranscript(true);
</script></body></html>"""


def create_app(runtime: System1Runtime) -> "FastAPI":
    """Build a debug-only app without making FastAPI a domain dependency."""
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
    except ImportError as error:
        raise RuntimeError("Install debug support: pip install -e '.[vad-debug]'") from error

    app = FastAPI(title="GhostPilot VAD Debug", docs_url=None, redoc_url=None)

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
        await emit(text.strip(), is_final=bool(payload.get("is_final")), turn_id=turn_id)
        return {"turn_id": turn_id, "is_final": bool(payload.get("is_final"))}

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        events = runtime.events.subscribe()
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
                    await websocket.send_json(
                        {"type": "event", "timestamp": time.time(), "event": asdict(event)}
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
                    command_task = asyncio.create_task(websocket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            event_task.cancel()
            command_task.cancel()
            runtime.events.unsubscribe(events)

    return app


async def run(host: str, port: int, device: str | None, threshold: float) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install debug support: pip install -e '.[vad-debug]'") from error

    audio_device: int | str | None = int(device) if device and device.isdigit() else device
    config = System1Config(
        audio=AudioConfig(device=audio_device),
        vad=VADConfig(speech_threshold=threshold),
    )
    runtime = System1Runtime(config=config)
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
    args = parser.parse_args()
    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("Install audio support: pip install -e '.[audio]'") from error
        print(sd.query_devices())
        return
    asyncio.run(run(args.host, args.port, args.device, args.threshold))


if __name__ == "__main__":
    main()
