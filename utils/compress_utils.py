import lzma
import dahuffman
import pickle
import numpy as np
import torch
import json
import struct
import warnings
from collections import Counter

try:
    from numba import njit
except ImportError:
    njit = None

import torch
import json
import struct
import numpy as np

def load_comp_web(filename):
    with open(filename, "rb") as f:
        # 1. Read the 4-byte header length (Little-endian unsigned int)
        header_size_bytes = f.read(4)
        if len(header_size_bytes) < 4:
            raise ValueError("File is too short or corrupted.")
        
        header_size = struct.unpack('<I', header_size_bytes)[0]

        # 2. Read and decode the JSON metadata
        json_bytes = f.read(header_size)
        web_metadata = json.loads(json_bytes.decode('utf-8'))

        # 3. Read the remaining binary payload
        binary_payload = f.read()

    # 4. Recursive function to rebuild the Python objects
    def reconstruct_node(node):
        if isinstance(node, dict):
            # Check if this dict is actually a metadata marker for binary data
            if "_type" in node:
                offset = int(node["offset"])
                length = int(node["length"])
                if (
                    offset < 0
                    or length < 0
                    or offset + length > len(binary_payload)
                ):
                    raise ValueError(
                        "Compressed binary marker is outside the payload: "
                        f"offset={offset}, length={length}, "
                        f"payload={len(binary_payload)}"
                    )
                
                if node["_type"] == "ndarray":
                    dtype = np.dtype(node["dtype"])
                    shape = tuple(node["shape"])
                    
                    # Extract EXACTLY the data bytes (ignoring the 4-byte padding we added for JS)
                    blob = binary_payload[offset:offset + length]
                    
                    # Reconstruct and reshape
                    arr = np.frombuffer(blob, dtype=dtype).copy()
                    return arr.reshape(shape)
                
                elif node["_type"] == "bytes":
                    return binary_payload[offset:offset + length]
            
            # Standard dictionary processing
            reconstructed_dict = {}
            for k, v in node.items():
                if k.lstrip('-').isdigit(): 
                    k = int(k)
                reconstructed_dict[k] = reconstruct_node(v)
            return reconstructed_dict

        elif isinstance(node, list):
            return [reconstruct_node(item) for item in node]
        else:
            return node

    return reconstruct_node(web_metadata)


def save_comp_web(filename, save_dict):
    binary_blobs = []
    current_offset = [0]

    def process_node(val):
        # 1. Handle PyTorch Tensors
        if isinstance(val, torch.Tensor):
            # .contiguous() is MANDATORY here to ensure JS arrays don't read memory out of order
            val = val.detach().cpu().contiguous().numpy()
        # 2. Handle NumPy Arrays
        if isinstance(val, np.ndarray):
            # Ensure Little-Endian (JS standard)
            if val.dtype.byteorder == '>':
                val = val.astype(val.dtype.newbyteorder('<'))                
            flat_bytes = val.tobytes()
            actual_length = len(flat_bytes)
            
            # JS-OPTIMIZATION: 4-Byte padding alignment
            padding_len = (4 - (actual_length % 4)) % 4
            padded_bytes = flat_bytes + (b'\x00' * padding_len)
            
            meta = {
                "_type": "ndarray",
                "dtype": str(val.dtype),
                "shape": val.shape,
                "offset": current_offset[0],
                "length": actual_length # Keep actual length for loader, offset accounts for padding
            }
            binary_blobs.append(padded_bytes)
            current_offset[0] += len(padded_bytes)
            return meta

        # 3. Handle Raw Bytes
        elif isinstance(val, (bytes, bytearray)):
            actual_length = len(val)
            
            # JS-OPTIMIZATION: 4-Byte padding alignment
            padding_len = (4 - (actual_length % 4)) % 4
            padded_bytes = val + (b'\x00' * padding_len)
            
            meta = {
                "_type": "bytes",
                "offset": current_offset[0],
                "length": actual_length
            }
            binary_blobs.append(padded_bytes)
            current_offset[0] += len(padded_bytes)
            return meta

        # 4. Handle Lists
        elif isinstance(val, list):
            return [process_node(v) for v in val]

        # 5. Handle Dictionaries
        elif isinstance(val, dict):
            cleaned_dict = {}
            for k, v in val.items():
                if isinstance(k, np.generic):
                    k = k.item()
                if not isinstance(k, (str, int, float, bool, type(None))):
                    k = str(k)
                cleaned_dict[k] = process_node(v)
            return cleaned_dict

        # 6. JSON Primitives
        else:
            if isinstance(val, np.generic):
                return val.item()
            return val

    web_metadata = process_node(save_dict)
    json_bytes = json.dumps(web_metadata, separators=(',', ':')).encode('utf-8')

    with open(filename, "wb") as f:
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(json_bytes)
        for blob in binary_blobs:
            f.write(blob)
            
    print(f"Exported to {filename}")
    print(f"-> Header: {len(json_bytes)} bytes | Binary Data: {current_offset[0]} bytes (Padded for JS)")



# Cluster labels are nonnegative, so -1 is a stable, JSON-safe EOF marker.
# Older web artifacts contain "_EOF" because save_comp_web stringified
# dahuffman's private EOF object.
HUFFMAN_EOF = -1
LEGACY_HUFFMAN_EOF = "_EOF"


if njit is not None:
    @njit
    def _huffman_decode_numba(encoded_array, code_keys, code_symbols):
        """Decode canonical (length, value) keys without Python loop state."""
        decoded = np.empty(encoded_array.size * 8, dtype=np.int64)
        decoded_count = 0
        code_value = 0
        code_length = 0

        for encoded_byte in encoded_array:
            byte_value = int(encoded_byte)
            for bit_shift in range(7, -1, -1):
                code_value = code_value * 2 + ((byte_value // (2 ** bit_shift)) % 2)
                code_length += 1
                code_key = (2 ** code_length) - 1 + code_value

                left = 0
                right = code_keys.size
                while left < right:
                    middle = (left + right) // 2
                    if code_keys[middle] < code_key:
                        left = middle + 1
                    else:
                        right = middle
                if left >= code_keys.size or code_keys[left] != code_key:
                    continue

                symbol = code_symbols[left]
                if symbol == -1:
                    return decoded[:decoded_count], 0, 0
                decoded[decoded_count] = symbol
                decoded_count += 1
                code_value = 0
                code_length = 0

        return decoded[:decoded_count], code_length, code_value
else:
    _huffman_decode_numba = None


def huffman_encode(data):
    # Avoid NumPy scalar comparisons with dahuffman's private EOF object.
    symbols = [int(value) for value in np.asarray(data).reshape(-1)]
    if not symbols:
        raise ValueError("Cannot Huffman-encode an empty label stream")
    codec = dahuffman.HuffmanCodec.from_frequencies(
        Counter(symbols),
        eof=HUFFMAN_EOF,
    )
    encoded_bytes = codec.encode(symbols)
    huffman_table = codec.get_code_table()
    return encoded_bytes, huffman_table

def huffman_decode(
    encoded_bytes,
    huffman_table,
    expected_count=None,
    stream_name="Huffman stream",
):
    """Decode integer labels from native or JSON-restored Huffman tables.

    Byte containers and JSON metadata are normalized before parsing the bit
    stream.  Decoding locally avoids depending on mutable module state inside
    dahuffman while retaining its final, possibly truncated EOF convention.
    """
    if HUFFMAN_EOF in huffman_table:
        # Keep the actual key object.  This matters for legacy sentinels whose
        # equality implementation is not safe to call from NumPy code.
        eof_symbol = next(
            symbol for symbol in huffman_table if symbol == HUFFMAN_EOF
        )
    elif LEGACY_HUFFMAN_EOF in huffman_table:
        # JSON loading creates a new string object, so comparing it with the
        # module constant by identity is not reliable.
        eof_symbol = next(
            symbol for symbol in huffman_table
            if symbol == LEGACY_HUFFMAN_EOF
        )
    else:
        # Legacy in-memory/pickle tables still contain dahuffman's private EOF
        # singleton. The reverse lookup below returns this exact object, so EOF
        # detection can use identity rather than its overloaded comparisons.
        eof_candidates = [
            symbol for symbol in huffman_table
            if repr(symbol) == "_EOF"
        ]
        if len(eof_candidates) != 1:
            raise ValueError("Huffman table does not contain a recognizable EOF symbol")
        eof_symbol = eof_candidates[0]

    normalized_table = {}
    seen_codes = set()
    for symbol, code in huffman_table.items():
        if not isinstance(code, (list, tuple)) or len(code) != 2:
            raise ValueError(f"Invalid Huffman code for symbol {symbol!r}: {code!r}")
        bit_count, value = code
        # Old comp.json files may contain these two fields as JSON strings.
        # Normalize once here so no string can reach the bit decoder.
        normalized_code = (int(bit_count), int(value))
        if (
            normalized_code[0] < 1
            or normalized_code[1] < 0
            or normalized_code[1] >= 2 ** normalized_code[0]
        ):
            raise ValueError(
                f"Invalid Huffman code for symbol {symbol!r}: {normalized_code!r}"
            )
        if normalized_code in seen_codes:
            raise ValueError(f"Duplicate Huffman code: {normalized_code!r}")
        seen_codes.add(normalized_code)
        normalized_table[symbol] = normalized_code

    # Normalize the complete byte stream before decoding. Iterating arbitrary
    # bytes-like containers directly is not portable: depending on the
    # container/version an element can be an int or a one-byte bytes object.
    # A flat uint8 array gives one canonical representation for both encoder
    # output and data restored from comp.json.
    try:
        encoded_array = np.frombuffer(encoded_bytes, dtype=np.uint8)
    except (TypeError, ValueError, BufferError):
        encoded_array = np.asarray(encoded_bytes, dtype=np.uint8).reshape(-1)

    code_entries = []
    for symbol, (bit_count, value) in normalized_table.items():
        decoded_symbol = -1 if symbol is eof_symbol else int(symbol)
        if decoded_symbol < 0 and symbol is not eof_symbol:
            raise ValueError(f"Huffman labels must be nonnegative, got {symbol!r}")
        code_entries.append(((2 ** bit_count) - 1 + value, decoded_symbol))
    code_entries.sort()
    code_keys = np.asarray([entry[0] for entry in code_entries], dtype=np.int64)
    code_symbols = np.asarray([entry[1] for entry in code_entries], dtype=np.int64)

    if _huffman_decode_numba is not None:
        decoded_symbols, code_length, code_value = _huffman_decode_numba(
            encoded_array,
            code_keys,
            code_symbols,
        )
    else:
        # Training installations include Numba through RAPIDS/cuML. Keep a
        # portable fallback for lightweight decode-only environments.
        lookup = {
            code: symbol for symbol, code in normalized_table.items()
        }
        decoded_symbols = []
        code_value = 0
        code_length = 0
        missing = object()
        for encoded_bit in np.unpackbits(encoded_array):
            code_value = code_value * 2 + int(encoded_bit)
            code_length += 1
            symbol = lookup.get((code_length, code_value), missing)
            if symbol is missing:
                continue
            if symbol is eof_symbol:
                code_value = 0
                code_length = 0
                break
            decoded_symbols.append(int(symbol))
            code_value = 0
            code_length = 0

    # HuffmanCodec.encode may end on a symbol boundary without writing EOF, or
    # may write only enough of the EOF code to fill the final byte.  Reject a
    # residual bit sequence unless it is exactly such an EOF prefix.
    if code_length:
        eof_bits, eof_value = normalized_table[eof_symbol]
        if (
            code_length >= eof_bits
            or code_value != eof_value // (2 ** (eof_bits - code_length))
        ):
            raise ValueError(f"{stream_name} ended with an invalid EOF prefix")

    decoded = np.asarray(decoded_symbols, dtype=np.uint16)

    if expected_count is not None:
        expected_count = int(expected_count)
        if decoded.size < expected_count:
            raise ValueError(
                f"{stream_name} decoded {decoded.size} labels; "
                f"expected {expected_count}"
            )
        if decoded.size > expected_count:
            warnings.warn(
                f"{stream_name} decoded {decoded.size - expected_count} "
                "trailing label(s) after the expected payload; ignoring "
                "storage-alignment padding",
                RuntimeWarning,
                stacklevel=2,
            )
            decoded = decoded[:expected_count]

    return decoded

def save_comp(filename, save_dict):
    with lzma.open(filename, "wb") as f:
        pickle.dump(save_dict, f)

def load_comp(filename):
    with lzma.open(filename, "rb") as f:
        save_dict = pickle.load(f)
    return save_dict

def write_storage(save_dict, byte, numG):
    for name in save_dict:
        if name in ('rot_feature_dim', 'dynamic_enabled', 'dynamic_gate_count'):
            continue
        if name == 'xyz':
            byte['xyz'] = len(save_dict['xyz'])
        elif "offset" in name:
            num_params = sum(v.size for v in save_dict[name].values())
            byte['MLPs'] += num_params*16/8
        elif 'MLP' in name:
            byte['MLPs'] += save_dict[name].shape[0]*16/8
        elif name in ('velocity', 'acceleration', 'time', 'duration', 'dynamic_gate_bits'):
            if 'dynamic' not in byte:
                byte['dynamic'] = 0
            if name == 'dynamic_gate_bits':
                byte['dynamic'] += save_dict[name].nbytes
            else:
                byte['dynamic'] += save_dict[name].size*16/8
        else:
            attr, comp = name.split('_', 1)
            if attr not in byte:
                byte[attr] = 0
            if 'code' in comp:
                for i in range(len(save_dict[name])):
                    byte[attr] += save_dict[name][i].shape[0]*save_dict[name][i].shape[1]*16/8
            else:
                for i in range(len(save_dict[name])):
                    byte[attr] += len(save_dict[name][i])
    byte['total'] = sum(v for k, v in byte.items() if k != 'total')
    dynamic_line = "\nDynamic: " + str(byte.get('dynamic', 0)) if 'dynamic' in byte else ""
    return "#G: " + str(numG) + "\nPosition: " + str(byte['xyz']) + "\nScale: " + str(byte['scale']) + "\nRotation: " + str(byte['rotation']) + "\nAppearance: " + str(byte['app']) + dynamic_line + "\nMLPs: " + str(byte['MLPs'])+  "\nopacity: " + str(byte['opacity'])+ "\nTotal: " + str(byte['total']) + "\n"

def splitBy3(a):
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer
