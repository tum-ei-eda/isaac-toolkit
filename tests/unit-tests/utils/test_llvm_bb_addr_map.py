"""LLVM 19–22 BB map wire formats and strict malformed-input rejection."""
import pytest

from isaac_toolkit.utils.llvm_bb_addr_map import decode_bb_addr_map


def uleb(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def record(version, features=0, address_size=4):
    header = bytes([version]) + features.to_bytes(2 if version >= 5 else 1, 'little')
    if features & 8:
        header += uleb(2)
    payload = b''
    for base, bb_id in ([(0x1000, 7), (0x2000, 9)] if features & 8 else [(0x1000, 7)]):
        payload += base.to_bytes(address_size, 'little') + uleb(1)
        payload += uleb(bb_id) + uleb(4)
        if features & 32:
            payload += uleb(2) + uleb(2) + uleb(2)
        payload += uleb(4) + uleb(1)
        if features & 64:
            payload += (0x1234567890abcdef).to_bytes(8, 'little')
    if features & 1:
        payload += uleb(1 << 40)
    for _ in range(2 if features & 8 else 1):
        if features & 2:
            payload += uleb(1 << 40)
        if features & 6 and features & 128:
            payload += uleb(44)
        if features & 4:
            payload += uleb(1) + uleb(7) + uleb(0x80000000)
            if features & 128:
                payload += uleb(33)
    return header + payload


@pytest.mark.parametrize('version,features', [(2, 0), (2, 7), (2, 15), (3, 32), (4, 64),
                                             (4, 111), (5, 0), (5, 239), (5, 128)])
@pytest.mark.parametrize('address_size', [4, 8])
def test_formats_and_following_record_alignment(version, features, address_size):
    raw = record(version, features, address_size)
    decoded = decode_bb_addr_map(raw + record(2, address_size=address_size), address_size)
    assert len(decoded) == 2
    assert decoded[0]['version'] == version
    size = 8 if features & 32 else 4
    assert decoded[0]['ranges'][0] == (0x1000, [(7, 0x1004, size)])
    if features & 8:
        assert decoded[0]['ranges'][1] == (0x2000, [(9, 0x2004, size)])
    assert decoded[1]['ranges'] == [(0x1000, [(7, 0x1004, 4)])]


@pytest.mark.parametrize('version,features', [(2, 0), (3, 32), (4, 111), (5, 239)])
def test_every_truncation_fails(version, features):
    data = record(version, features)
    for end in range(1, len(data)):
        with pytest.raises(ValueError):
            decode_bb_addr_map(data[:end], 4)


@pytest.mark.parametrize('raw', [b'\x01', b'\x06', b'\x05\x00\x01'])
def test_unknown_format_or_feature_fails(raw):
    with pytest.raises(NotImplementedError):
        decode_bb_addr_map(raw, 4)


@pytest.mark.parametrize('version,features', [(2, 32), (3, 64), (4, 128), (5, 16)])
def test_incompatible_features_fail(version, features):
    with pytest.raises(ValueError):
        decode_bb_addr_map(record(version, features), 4)


def test_duplicates_and_overflow_fail():
    header = b'\x02\x00' + (0x1000).to_bytes(4, 'little') + b'\x02'
    with pytest.raises(ValueError, match='Duplicate BB ID'):
        decode_bb_addr_map(header + b'\x07\x00\x04\x00' * 2, 4)
    with pytest.raises(ValueError, match='Oversized ULEB128'):
        decode_bb_addr_map(header + b'\xff' * 5, 4)


def test_empty_multi_range_fails():
    with pytest.raises(ValueError, match='no address ranges'):
        decode_bb_addr_map(b'\x02\x08\x00', 4)
