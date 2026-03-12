from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np


class VastProtocolError(RuntimeError):
    def __init__(self, message: str, *, result_code: int | None = None, error_code: int | None = None) -> None:
        super().__init__(message)
        self.result_code = result_code
        self.error_code = error_code


@dataclass
class VastResponse:
    result_code: int
    payload: bytes


class VastClient:
    GETINFO = 1
    GETVIEWCOORDINATES = 8
    GETSELECTEDSEGMENTNR = 17
    GETSELECTEDLAYERNR = 19
    GETSEGIMAGERAW = 20
    GETEMIMAGERAW = 30
    GETEMIMAGERAWIMMEDIATE = 31
    REFRESHLAYERREGION = 32
    SETSEGIMAGERAW = 50
    SETSEGIMAGERLE = 51
    GETAPILAYERSENABLED = 101
    SETAPILAYERSENABLED = 102
    SETSELECTEDAPILAYERNR = 104
    GETCURRENTUISTATE = 110

    def __init__(self, host: str = "127.0.0.1", port: int = 22081, timeout_s: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None

    def __enter__(self) -> "VastClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._socket = sock

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def get_info(self) -> dict[str, list[int | float]]:
        response = self._send_message(self.GETINFO, b"")
        ints, uints, doubles, texts, uint64s = self._parse_typed_payload(response.payload)
        return {
            "ints": ints,
            "uints": uints,
            "doubles": doubles,
            "texts": texts,
            "uint64s": uint64s,
        }

    def get_view_coordinates(self) -> tuple[int, int, int]:
        response = self._send_message(self.GETVIEWCOORDINATES, b"")
        ints, _, _, _, _ = self._parse_typed_payload(response.payload)
        if len(ints) < 3:
            raise VastProtocolError("GETVIEWCOORDINATES returned an unexpected payload")
        return ints[0], ints[1], ints[2]

    def get_selected_segment_nr(self) -> int:
        response = self._send_message(self.GETSELECTEDSEGMENTNR, b"")
        _, uints, _, _, _ = self._parse_typed_payload(response.payload)
        if len(uints) != 1:
            raise VastProtocolError("GETSELECTEDSEGMENTNR returned an unexpected payload")
        return uints[0]

    def get_selected_layer_info(self) -> tuple[int, int, int]:
        response = self._send_message(self.GETSELECTEDLAYERNR, b"")
        ints, _, _, _, _ = self._parse_typed_payload(response.payload)
        if len(ints) == 3:
            return ints[0], ints[1], ints[2]
        if len(ints) == 5:
            return ints[0], ints[1], ints[3]
        raise VastProtocolError("GETSELECTEDLAYERNR returned an unexpected payload")

    def set_selected_api_layer_nr(self, layer_nr: int) -> None:
        self._send_message(self.SETSELECTEDAPILAYERNR, self._encode_int32_values([layer_nr]))

    def get_api_layers_enabled(self) -> int:
        response = self._send_message(self.GETAPILAYERSENABLED, b"")
        _, uints, _, _, _ = self._parse_typed_payload(response.payload)
        if len(uints) != 1:
            raise VastProtocolError("GETAPILAYERSENABLED returned an unexpected payload")
        return uints[0]

    def set_api_layers_enabled(self, enabled: bool) -> None:
        self._send_message(self.SETAPILAYERSENABLED, self._encode_uint32_values([1 if enabled else 0]))

    def get_current_ui_state(self) -> dict[str, int]:
        response = self._send_message(self.GETCURRENTUISTATE, b"")
        ints, uints, _, _, _ = self._parse_typed_payload(response.payload)
        if len(ints) != 9 or len(uints) != 5:
            raise VastProtocolError("GETCURRENTUISTATE returned an unexpected payload")
        flags = uints[0]
        return {
            "mousecoordsx": ints[0],
            "mousecoordsy": ints[1],
            "lastleftclickx": ints[2],
            "lastleftclicky": ints[3],
            "lastleftreleasex": ints[4],
            "lastleftreleasey": ints[5],
            "mousecoordsz": ints[6],
            "clientwindowwidth": ints[7],
            "clientwindowheight": ints[8],
            "reservedflag": flags & 1,
            "lbuttondown": (flags >> 1) & 1,
            "rbuttondown": (flags >> 2) & 1,
            "mbuttondown": (flags >> 3) & 1,
            "ctrlpressed": (flags >> 4) & 1,
            "shiftpressed": (flags >> 5) & 1,
            "deletepressed": (flags >> 6) & 1,
            "spacepressed": (flags >> 7) & 1,
            "spacewaspressed": (flags >> 8) & 1,
            "uimode": uints[1],
            "hoversegmentnr": uints[2],
            "miplevel": uints[3],
            "paintcursordiameter": uints[4],
        }

    def get_em_image(
        self,
        layer_nr: int,
        miplevel: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
    ) -> np.ndarray:
        response = self._fetch_em_image_response(layer_nr, miplevel, minx, maxx, miny, maxy, minz, maxz)
        width = maxx - minx + 1
        height = maxy - miny + 1
        depth = maxz - minz + 1
        expected_pixels = width * height * depth
        bytes_per_pixel = len(response.payload) // expected_pixels
        if expected_pixels <= 0 or bytes_per_pixel not in {1, 3, 4, 8}:
            raise VastProtocolError("GETEMIMAGERAW returned an unexpected payload size")
        if bytes_per_pixel != 1 or depth != 1:
            raise VastProtocolError("This bridge currently supports single-slice grayscale EM data only")
        image = np.frombuffer(response.payload, dtype=np.uint8).reshape((width, height)).T.copy()
        return image

    def _fetch_em_image_response(
        self,
        layer_nr: int,
        miplevel: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
    ) -> VastResponse:
        candidate_layers = [layer_nr]
        if layer_nr >= 0:
            candidate_layers.append(layer_nr + 1)
        last_exc: VastProtocolError | None = None
        for candidate_layer in candidate_layers:
            payload = self._encode_uint32_values([candidate_layer, miplevel, minx, maxx, miny, maxy, minz, maxz])
            try:
                return self._send_message(self.GETEMIMAGERAW, payload)
            except VastProtocolError as exc:
                last_exc = exc
                if exc.error_code != 3:
                    raise
                immediate_payload = self._encode_uint32_values(
                    [candidate_layer, miplevel, minx, maxx, miny, maxy, minz, maxz, 1]
                )
                try:
                    return self._send_message(self.GETEMIMAGERAWIMMEDIATE, immediate_payload)
                except VastProtocolError as exc_immediate:
                    last_exc = exc_immediate
                    if exc_immediate.error_code != 3:
                        raise
                    continue
        assert last_exc is not None
        raise last_exc

    def get_seg_image(
        self,
        miplevel: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
    ) -> np.ndarray:
        payload = self._encode_uint32_values([miplevel, minx, maxx, miny, maxy, minz, maxz])
        response = self._send_message(self.GETSEGIMAGERAW, payload)
        width = maxx - minx + 1
        height = maxy - miny + 1
        depth = maxz - minz + 1
        if depth != 1:
            raise VastProtocolError("This bridge currently supports single-slice segmentation writes only")
        expected_values = width * height * depth
        if len(response.payload) != expected_values * 2:
            raise VastProtocolError("GETSEGIMAGERAW returned an unexpected payload size")
        image = np.frombuffer(response.payload, dtype="<u2").reshape((width, height)).T.copy()
        return image

    def set_seg_image_rle(
        self,
        miplevel: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
        segimage: np.ndarray,
    ) -> None:
        self._validate_segimage_shape(segimage, minx, maxx, miny, maxy)
        if segimage.dtype != np.uint16:
            segimage = segimage.astype(np.uint16, copy=False)
        raw = np.ascontiguousarray(segimage.T).reshape(-1)
        encoded = self._encode_rle(raw)
        payload = self._encode_uint32_values([miplevel, minx, maxx, miny, maxy, minz, maxz]) + self._encode_data_block(encoded)
        try:
            self._send_message(self.SETSEGIMAGERLE, payload)
        except VastProtocolError as exc:
            if exc.error_code != 3:
                raise
            self.set_seg_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz, segimage)

    def set_seg_image_raw(
        self,
        miplevel: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
        segimage: np.ndarray,
    ) -> None:
        self._validate_segimage_shape(segimage, minx, maxx, miny, maxy)
        if segimage.dtype != np.uint16:
            segimage = segimage.astype(np.uint16, copy=False)
        raw = np.ascontiguousarray(segimage.T).reshape(-1).astype("<u2", copy=False).tobytes(order="C")
        payload = self._encode_uint32_values([miplevel, minx, maxx, miny, maxy, minz, maxz]) + self._encode_data_block(raw)
        self._send_message(self.SETSEGIMAGERAW, payload)

    @staticmethod
    def _validate_segimage_shape(segimage: np.ndarray, minx: int, maxx: int, miny: int, maxy: int) -> None:
        expected_shape = (maxy - miny + 1, maxx - minx + 1)
        if segimage.shape != expected_shape:
            raise ValueError(f"segimage shape {segimage.shape} does not match expected shape {expected_shape}")

    def refresh_layer_region(
        self,
        layer_nr: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
    ) -> None:
        payload = self._encode_uint32_values([layer_nr, minx, maxx, miny, maxy, minz, maxz])
        self._send_message(self.REFRESHLAYERREGION, payload)

    def _send_message(self, message_nr: int, payload: bytes) -> VastResponse:
        if self._socket is None:
            self.connect()
        assert self._socket is not None
        header = b"VAST" + struct.pack("<Q", len(payload) + 4) + struct.pack("<I", message_nr)
        self._socket.sendall(header + payload)
        raw_header = self._recv_exact(16)
        if raw_header[:4] != b"VAST":
            raise VastProtocolError("Invalid VAST response header")
        payload_length_plus_result = struct.unpack("<Q", raw_header[4:12])[0]
        result_code = struct.unpack("<i", raw_header[12:16])[0]
        payload_length = max(int(payload_length_plus_result) - 4, 0)
        response_payload = self._recv_exact(payload_length) if payload_length else b""
        if result_code != 1:
            error_detail = ""
            try:
                _, uints, _, _, _ = self._parse_typed_payload(response_payload)
                if uints:
                    error_detail = f" (VAST error {uints[0]})"
            except Exception:
                error_detail = ""
            error_code = None
            try:
                _, uints, _, _, _ = self._parse_typed_payload(response_payload)
                if uints:
                    error_code = uints[0]
            except Exception:
                error_code = None
            raise VastProtocolError(
                f"VAST command {message_nr} failed with result code {result_code}{error_detail}",
                result_code=result_code,
                error_code=error_code,
            )
        return VastResponse(result_code=result_code, payload=response_payload)

    def _recv_exact(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            assert self._socket is not None
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise VastProtocolError("Connection closed while reading VAST response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _parse_typed_payload(payload: bytes) -> tuple[list[int], list[int], list[float], list[str], list[int]]:
        ints: list[int] = []
        uints: list[int] = []
        doubles: list[float] = []
        texts: list[str] = []
        uint64s: list[int] = []
        pos = 0
        while pos < len(payload):
            type_id = payload[pos]
            pos += 1
            if type_id == 1:
                if pos + 4 > len(payload):
                    break
                uints.append(struct.unpack("<I", payload[pos : pos + 4])[0])
                pos += 4
            elif type_id == 2:
                if pos + 8 > len(payload):
                    break
                doubles.append(struct.unpack("<d", payload[pos : pos + 8])[0])
                pos += 8
            elif type_id == 3:
                end = payload.find(b"\x00", pos)
                if end == -1:
                    break
                texts.append(payload[pos:end].decode("utf-8", errors="replace"))
                pos = end + 1
            elif type_id == 4:
                if pos + 4 > len(payload):
                    break
                ints.append(struct.unpack("<i", payload[pos : pos + 4])[0])
                pos += 4
            elif type_id == 6:
                if pos + 8 > len(payload):
                    break
                uint64s.append(struct.unpack("<Q", payload[pos : pos + 8])[0])
                pos += 8
            else:
                break
        return ints, uints, doubles, texts, uint64s

    @staticmethod
    def _encode_rle(data: np.ndarray) -> bytes:
        if data.ndim != 1:
            raise ValueError("RLE input must be flat")
        if data.size == 0:
            return b""
        pairs: list[int] = []
        current = int(data[0])
        count = 1
        for value in data[1:]:
            value_int = int(value)
            if value_int == current and count < 65535:
                count += 1
            else:
                pairs.extend([current, count])
                current = value_int
                count = 1
        pairs.extend([current, count])
        return np.asarray(pairs, dtype=np.uint16).tobytes(order="C")

    @staticmethod
    def _encode_uint32_values(values: list[int]) -> bytes:
        return b"".join(b"\x01" + struct.pack("<I", int(value)) for value in values)

    @staticmethod
    def _encode_int32_values(values: list[int]) -> bytes:
        return b"".join(b"\x04" + struct.pack("<i", int(value)) for value in values)

    @staticmethod
    def _encode_data_block(data: bytes) -> bytes:
        return b"\x05" + struct.pack("<I", len(data)) + data
