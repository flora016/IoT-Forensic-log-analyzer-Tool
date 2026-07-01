# =============================================================================
#  parser/pcap_features.py
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from config import CFG
from scapy.all import PcapReader

# ── Library detection with clear messages ────────────────────────────────────
try:
    from scapy.all import rdpcap, IP, TCP, UDP
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

try:
    import pyshark
    PYSHARK_OK = True
except ImportError:
    PYSHARK_OK = False


def _check_libraries():
    """Print clear status of available parsing libraries."""
    print("  [PCAP] Library status:")
    print(f"         scapy    : {'OK' if SCAPY_OK   else 'NOT INSTALLED  ->  pip install scapy'}")
    print(f"         pyshark  : {'OK' if PYSHARK_OK else 'NOT INSTALLED  ->  pip install pyshark'}")
    if not SCAPY_OK and not PYSHARK_OK:
        print("  [PCAP] ERROR: No PCAP library available. Install scapy: pip install scapy")


# ── Scapy parser ─────────────────────────────────────────────────────────────

def _parse_pcap_scapy(filepath: Path) -> pd.DataFrame:
    packets = rdpcap(str(filepath))
    if not packets:
        print(f"  [PCAP] scapy: 0 packets read from {filepath.name}")
        return pd.DataFrame()

    records = []
    for pkt in packets:
        try:
            ts     = float(pkt.time)
            length = len(pkt)
            proto  = "Other"
            src    = dst = "N/A"
            is_syn = 0
            if IP in pkt:
                src = pkt[IP].src
                dst = pkt[IP].dst
                if TCP in pkt:
                    proto  = "TCP"
                    flags  = int(pkt[TCP].flags)
                    is_syn = 1 if (flags & 0x02 and not flags & 0x10) else 0
                elif UDP in pkt:
                    proto = "UDP"
            records.append({
                "ts": pd.Timestamp(ts, unit="s", tz="UTC"),
                "src": src, "dst": dst,
                "protocol": proto, "length": length, "is_syn": is_syn,
            })
        except Exception:
            continue
    print(f"  [PCAP] scapy: {len(records)} packets parsed from {filepath.name}")
    return pd.DataFrame(records)


# ── Pyshark parser (fallback) ─────────────────────────────────────────────────

def _parse_pcap_pyshark(filepath: Path) -> pd.DataFrame:
    cap     = pyshark.FileCapture(str(filepath), keep_packets=False)
    records = []
    for pkt in cap:
        try:
            ts     = pd.Timestamp(float(pkt.sniff_timestamp), unit="s", tz="UTC")
            proto  = pkt.transport_layer or "Other"
            length = int(pkt.length)
            src    = pkt.ip.src if hasattr(pkt, "ip") else "N/A"
            dst    = pkt.ip.dst if hasattr(pkt, "ip") else "N/A"
            is_syn = 0
            if hasattr(pkt, "tcp"):
                try:
                    flags  = int(pkt.tcp.flags, 16)
                    is_syn = 1 if (flags & 0x02 and not flags & 0x10) else 0
                except Exception:
                    pass
            records.append({
                "ts": ts, "src": src, "dst": dst,
                "protocol": proto, "length": length, "is_syn": is_syn,
            })
        except Exception:
            continue
    cap.close()
    print(f"  [PCAP] pyshark: {len(records)} packets parsed from {filepath.name}")
    return pd.DataFrame(records)


# ── Time-bin aggregation ─────────────────────────────────────────────────────

def _bin_network_features(pkt_df: pd.DataFrame, bin_sec: int) -> pd.DataFrame:
    if pkt_df.empty:
        return pd.DataFrame()

    pkt_df = pkt_df.sort_values("ts").copy()
    pkt_df["bin"] = pkt_df["ts"].dt.floor(f"{bin_sec}s")

    def entropy(s: pd.Series) -> float:
        counts = s.value_counts(normalize=True)
        return float(-(counts * np.log2(counts + 1e-9)).sum())

    agg = pkt_df.groupby("bin").agg(
        pkt_count      = ("length", "count"),
        total_bytes    = ("length", "sum"),
        unique_src_ips = ("src",    "nunique"),
        tcp_syn_count  = ("is_syn", "sum"),
    ).reset_index()

    proto_ent = pkt_df.groupby("bin")["protocol"].apply(entropy).reset_index()
    proto_ent.columns = ["bin", "proto_entropy"]
    agg = agg.merge(proto_ent, on="bin", how="left")

    agg["pkt_rate"]     = agg["pkt_count"]    / bin_sec
    agg["bytes_rate"]   = agg["total_bytes"]   / bin_sec
    agg["tcp_syn_rate"] = agg["tcp_syn_count"] / bin_sec
    agg["dos_flag"]     = (agg["pkt_rate"]     > CFG["dos_pkt_rate"]).astype(int)
    agg["syn_flag"]     = (agg["tcp_syn_rate"] > CFG["syn_rate_threshold"]).astype(int)

    rows = []
    for _, r in agg.iterrows():
        flags = []
        if r["dos_flag"]: flags.append("DoS burst")
        if r["syn_flag"]: flags.append("SYN flood")
        event = "Network: " + (", ".join(flags) if flags else "Normal traffic")
        rows.append({
            "timestamp"     : r["bin"],
            "source"        : "network",
            "event"         : event,
            "value"         : r["pkt_rate"],
            "pkt_rate"      : r["pkt_rate"],
            "bytes_rate"    : r["bytes_rate"],
            "unique_src_ips": r["unique_src_ips"],
            "tcp_syn_rate"  : r["tcp_syn_rate"],
            "proto_entropy" : r["proto_entropy"],
            "dos_flag"      : r["dos_flag"],
            "syn_flag"      : r["syn_flag"],
        })
    return pd.DataFrame(rows)


# ── Public API ───────────────────────────────────────────────────────────────

def parse_pcap(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"  [PCAP] File not found: {filepath}")
        return pd.DataFrame()

    print(f"  [PCAP] Parsing: {filepath.name}  ({filepath.stat().st_size / 1024:.1f} KB)")
    _check_libraries()

    pkt_df = pd.DataFrame()

    if SCAPY_OK:
        try:
            pkt_df = _parse_pcap_scapy(filepath)
        except Exception as e:
            print(f"  [PCAP] scapy failed: {e}")

    if pkt_df.empty and PYSHARK_OK:
        try:
            pkt_df = _parse_pcap_pyshark(filepath)
        except Exception as e:
            print(f"  [PCAP] pyshark failed: {e}")

    if pkt_df.empty:
        print("  [PCAP] No packets extracted. Check library install or file format.")
        return pd.DataFrame()

    binned = _bin_network_features(pkt_df, CFG["pcap_bin_seconds"])
    print(f"  [PCAP] {len(pkt_df)} packets  ->  {len(binned)} time bins ({CFG['pcap_bin_seconds']}s each)")
    return binned


def parse_pcap_dir(dirpath: str | Path) -> pd.DataFrame:
    dirpath = Path(dirpath)

    # Support both .pcap and .pcapng (Wireshark default in newer versions)
    files = sorted(list(dirpath.glob("*.pcap")) + list(dirpath.glob("*.pcapng")))

    if not files:
        print(f"  [PCAP] No .pcap or .pcapng files found in: {dirpath}")
        print(f"         Place your Wireshark capture file in:  {dirpath}/")
        return pd.DataFrame()

    print(f"  [PCAP] Found {len(files)} file(s): {[f.name for f in files]}")
    frames = [parse_pcap(f) for f in files]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    print(f"  [PCAP] Total bins after merge: {len(result)}")
    return result