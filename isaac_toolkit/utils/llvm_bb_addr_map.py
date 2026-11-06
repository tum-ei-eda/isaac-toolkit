"""Bounded decoder for linked LLVM BB address maps (LLVM 19–22).

Layout follows llvm/lib/Object/ELF.cpp, decodeBBAddrMap, versions 2–5.
Addresses require relocations to have been applied by the linker.
"""
import io


def decode_bb_addr_map(data: bytes, address_size: int):
    """Return function records with ranges and fixed block IDs.

    Optional callsite, hash and PGO payloads are consumed to keep subsequent
    records aligned. Maps without BB entries cannot support profile matching.
    """
    if address_size not in (4, 8):
        raise ValueError("BB map address size must be 4 or 8")
    reader = io.BytesIO(data)

    def fixed(size):
        raw = reader.read(size)
        if len(raw) != size:
            raise ValueError(f"Truncated BB address map at offset {reader.tell()}")
        return int.from_bytes(raw, "little")

    def uleb(bits=32):
        value = 0
        for shift in range(0, bits, 7):
            byte = fixed(1)
            value |= (byte & 0x7f) << shift
            if value >= 1 << bits:
                raise ValueError("Oversized ULEB128 in BB address map")
            if not byte & 0x80:
                return value
        raise ValueError("Oversized ULEB128 in BB address map")

    def count(bits=32):
        value = uleb(bits)
        # Every element below consumes at least one byte. Bound loops before
        # allocating or iterating over a corrupted untrusted count.
        if value > len(data) - reader.tell():
            raise ValueError("BB map count exceeds remaining payload")
        return value

    records = []
    while reader.tell() < len(data):
        version = fixed(1)
        if version not in (2, 3, 4, 5):
            raise NotImplementedError(f"LLVM_BB_ADDR_MAP version {version}")
        features = fixed(2 if version >= 5 else 1)
        if features & ~0xff:
            raise NotImplementedError(f"Unsupported BB address map features: {features:#x}")
        if ((features & 0x20 and version < 3) or
                (features & 0x40 and version < 4) or
                (features & 0x80 and version < 5)):
            raise ValueError("BB map feature is incompatible with its version")
        if features & 0x10:
            raise ValueError("BB map omits the entries required for profile matching")
        multi = bool(features & 8)
        num_ranges = count() if multi else 1
        if num_ranges == 0:
            raise ValueError("BB map has no address ranges")
        ranges = []
        ids = set()
        for _ in range(num_ranges):
            base = fixed(address_size)
            num_blocks = count()
            blocks = []
            previous_end = 0
            for _ in range(num_blocks):
                bb_id = uleb()
                if bb_id in ids:
                    raise ValueError(f"Duplicate BB ID {bb_id} in BB map function")
                ids.add(bb_id)
                offset = previous_end + uleb()
                callsite_size = 0
                if features & 0x20:
                    for _ in range(count()):
                        callsite_size += uleb()
                size = callsite_size + uleb()
                metadata = uleb()
                if metadata & ~0x1f:
                    raise ValueError(f"Unknown BB metadata: {metadata:#x}")
                if features & 0x40:
                    fixed(8)
                previous_end = offset + size
                if previous_end >= 1 << 32 or base + previous_end > 1 << (address_size * 8):
                    raise ValueError("BB map address/offset overflow")
                blocks.append((bb_id, base + offset, size))
            ranges.append((base, blocks))
        if features & 1:
            uleb(64)  # Function entry count.
        # Match LLVM's hasPGOAnalysisBBData(): PostLinkCfg alone does not
        # introduce a per-block payload, but adds fields to BBFreq/BrProb.
        if features & 6:
            for _ in range(len(ids)):
                if features & 2:
                    uleb(64)
                if features & 0x80:
                    uleb()
                if features & 4:
                    for _ in range(count(64)):
                        uleb()  # Successor BB ID.
                        uleb()  # Raw branch probability.
                        if features & 0x80:
                            uleb()
        records.append({"version": version, "features": features, "ranges": ranges})
    return records
