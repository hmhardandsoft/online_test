"""Low-frequency ETW anchors; no work inside the stimulus frame loop.

The string-only ETW provider is deliberately limited to diagnostic markers.
It is not a GPU timer. EventWriteString status zero does NOT prove a collector
received an event: consult provider_enabled and verify markers in the ETL.
"""
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import uuid


PROVIDER_GUID = '8c88351d-6ab9-480a-83a3-96a7df776154'


class WindowsMarker:
    def __init__(self):
        self.handle = ctypes.c_ulonglong()
        self.api = None
        self.error = None
        self.frequency = None
        try:
            if os.name != 'nt':
                raise OSError('ETW is available only on Windows')
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
            self.kernel.QueryPerformanceCounter.restype = ctypes.c_int
            self.kernel.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
            self.kernel.QueryPerformanceFrequency.restype = ctypes.c_int
            value = ctypes.c_longlong()
            if not self.kernel.QueryPerformanceFrequency(ctypes.byref(value)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.frequency = value.value
            api = ctypes.WinDLL('advapi32', use_last_error=True)
            api.EventRegister.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_ulonglong)]
            api.EventRegister.restype = ctypes.c_ulong
            api.EventWriteString.argtypes = [ctypes.c_ulonglong, ctypes.c_ubyte,
                                             ctypes.c_ulonglong, ctypes.c_wchar_p]
            api.EventWriteString.restype = ctypes.c_ulong
            api.EventProviderEnabled.argtypes = [ctypes.c_ulonglong, ctypes.c_ubyte, ctypes.c_ulonglong]
            api.EventProviderEnabled.restype = ctypes.c_ubyte
            api.EventUnregister.argtypes = [ctypes.c_ulonglong]
            api.EventUnregister.restype = ctypes.c_ulong
            guid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(PROVIDER_GUID).bytes_le)
            status = api.EventRegister(guid, None, None, ctypes.byref(self.handle))
            if status:
                raise OSError(status, 'EventRegister failed')
            self.api = api
        except Exception as exc:
            self.error = str(exc)

    def ticks(self):
        if self.frequency is None:
            return None
        value = ctypes.c_longlong()
        if not self.kernel.QueryPerformanceCounter(ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value

    def enabled(self):
        return bool(self.api and self.api.EventProviderEnabled(self.handle, 4, 1))

    def emit(self, message):
        return int(self.api.EventWriteString(self.handle, 4, 1, message)) if self.api else None

    def close(self):
        if self.api:
            self.api.EventUnregister(self.handle)
            self.api = None


class RenderTraceContext:
    def __init__(self, output, enabled=True, marker_factory=WindowsMarker):
        self.enabled = enabled
        self.output = Path(output)
        self.marker = marker_factory() if enabled else None
        self.data = dict(schema_version=1, run_id=self.output.name,
                         pid=os.getpid(), main_thread_native_id=threading.get_native_id(),
                         python=sys.version, executable=sys.executable,
                         platform=platform.platform(), provider_guid=PROVIDER_GUID,
                         workers=[], anchors=[], errors=[], display=None,
                         clock_info=vars(time.get_clock_info('perf_counter')),
                         qpc_frequency_hz=self.marker.frequency if self.marker else None,
                         etw_registration_error=self.marker.error if self.marker else None,
                         physical_display_timing_verified=False)

    def anchor(self, name):
        if not self.enabled:
            return
        try:
            # Separate QPC/UTC brackets preserve the uncertainty of cross-clock reads.
            q0 = time.perf_counter()
            ticks = self.marker.ticks()
            q1 = time.perf_counter()
            u0 = time.perf_counter()
            utc = time.time_ns()
            u1 = time.perf_counter()
            payload = dict(run_id=self.data['run_id'], pid=self.data['pid'],
                           tid=threading.get_native_id(), name=name)
            message = 'SSVEP_RENDER ' + json.dumps(payload, separators=(',', ':'))
            enabled = self.marker.enabled()
            before = time.perf_counter()
            status = self.marker.emit(message)
            after = time.perf_counter()
            self.data['anchors'].append(dict(**payload, provider_enabled=enabled,
                event_status=status, emit_before=before, emit_after=after,
                emit_bracket_ms=(after-before)*1000, qpc_ticks=ticks,
                qpc_perf_before=q0, qpc_perf_after=q1,
                utc_ns=utc, utc_perf_before=u0, utc_perf_after=u1))
        except Exception as exc:
            self.data['errors'].append(f'anchor {name}: {exc}')

    def display(self, glfw, window):
        if not self.enabled or glfw is None:
            return
        try:
            from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
            def string(token):
                value = glGetString(token)
                return value.decode('utf-8', 'replace') if isinstance(value, bytes) else str(value)
            monitor = glfw.get_window_monitor(window)
            mode = glfw.get_video_mode(monitor) if monitor else None
            self.data['display'] = dict(gl_vendor=string(GL_VENDOR), gl_renderer=string(GL_RENDERER),
                gl_version=string(GL_VERSION), glfw_version=glfw.get_version_string().decode('utf-8', 'replace'),
                framebuffer_size=list(glfw.get_framebuffer_size(window)),
                fullscreen=bool(monitor), refresh_rate=mode.refresh_rate if mode else None,
                swap_interval_requested=1,
                note='Requested swap interval is not proof of driver/display presentation behavior')
        except Exception as exc:
            self.data['errors'].append(f'display: {exc}')

    def workers(self, workers):
        if self.enabled:
            self.data['workers'] = [dict(name=p.name, pid=p.pid) for p in workers]

    def save(self):
        if self.enabled:
            try:
                (self.output/'render_trace_context.json').write_text(
                    json.dumps(self.data, indent=2), encoding='utf-8')
            except Exception as exc:
                # Supplementary diagnostics must not interrupt acquisition or cleanup.
                self.data['errors'].append(f'save: {exc}')
                print(f'[RenderTrace] Could not save context: {exc}', flush=True)

    def close(self):
        if self.enabled:
            self.anchor('run_stop')
            self.marker.close()
            self.save()
