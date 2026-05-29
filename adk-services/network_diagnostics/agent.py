"""Network Diagnostics ADK Agent — exposed via A2A on port 8001.

Start with: python adk-services/network_diagnostics/agent.py
Agent card will be at: http://localhost:8001/.well-known/agent-card.json
"""
import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

DB_PATH = os.environ.get(
    "DB_PATH",
    str(Path(__file__).parent.parent.parent / "data" / "telecom_ops.db"),
)
ADK_MODEL = os.environ.get("ADK_MODEL", "gemini-2.0-flash-exp")
PORT = 8001


def _db():
    return sqlite3.connect(DB_PATH)


async def check_tower_status(tower_id: str) -> str:
    """Return status, technology, performance metrics, and any open incident for a tower.

    Args:
        tower_id: The tower identifier (e.g. TX-512).
    """
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.tower_id, t.region, t.city, t.technology, t.status,
               p.avg_latency_ms, p.packet_loss_pct, p.throughput_mbps,
               p.signal_strength_dbm, p.recorded_at,
               i.incident_id, i.severity AS inc_sev, i.description AS inc_desc,
               i.eta_hours, i.status AS inc_status
        FROM network_towers t
        LEFT JOIN tower_performance p ON t.tower_id = p.tower_id
        LEFT JOIN open_incidents i ON t.tower_id = i.tower_id
        WHERE t.tower_id = ?
        ORDER BY p.recorded_at DESC
        LIMIT 1
        """,
        (tower_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return json.dumps({"error": f"Tower {tower_id} not found."})

    return json.dumps({
        "tower_id": row[0],
        "region": row[1],
        "city": row[2],
        "technology": row[3],
        "status": row[4],
        "performance": {
            "avg_latency_ms": row[5],
            "packet_loss_pct": row[6],
            "throughput_mbps": row[7],
            "signal_strength_dbm": row[8],
            "recorded_at": row[9],
        },
        "open_incident": {
            "incident_id": row[10],
            "severity": row[11],
            "description": row[12],
            "eta_hours": row[13],
            "status": row[14],
        } if row[10] else None,
    })


async def run_connectivity_diagnostics(tower_id: str, symptom: str) -> str:
    """Diagnose connectivity issues at a tower and return recommendations.

    Args:
        tower_id: The tower identifier (e.g. TX-512).
        symptom: Reported issue description (e.g. '5G drops', 'slow speeds').
    """
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.status, t.technology,
               p.avg_latency_ms, p.packet_loss_pct, p.throughput_mbps, p.signal_strength_dbm,
               i.incident_id, i.severity, i.description, i.eta_hours
        FROM network_towers t
        LEFT JOIN tower_performance p ON t.tower_id = p.tower_id
        LEFT JOIN open_incidents i ON t.tower_id = i.tower_id
        WHERE t.tower_id = ?
        ORDER BY p.recorded_at DESC
        LIMIT 1
        """,
        (tower_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return json.dumps({"error": f"Tower {tower_id} not found."})

    status, tech, latency, pkt_loss, throughput, signal, inc_id, inc_sev, inc_desc, eta = row
    recommendations = []

    if status in ("OUTAGE", "MAINTENANCE"):
        recommendations.append(f"Tower is currently {status}. Service unavailable until restoration.")
    if status == "DEGRADED":
        recommendations.append("Tower is DEGRADED — performance is reduced.")
    if pkt_loss is not None and pkt_loss > 5.0:
        recommendations.append(
            f"High packet loss ({pkt_loss:.1f}%) — likely congestion or hardware fault. NOC inspection recommended."
        )
    if latency is not None and latency > 100:
        recommendations.append(f"Elevated latency ({latency:.0f} ms) — check backhaul fiber or transport links.")
    if signal is not None and signal < -90:
        recommendations.append(f"Weak signal ({signal:.0f} dBm) — customer may be at coverage edge or indoors.")
    if throughput is not None and throughput < 10:
        recommendations.append(f"Low throughput ({throughput:.1f} Mbps) — possible congestion or hardware degradation.")
    if inc_id:
        recommendations.append(
            f"Active {inc_sev} incident {inc_id}: {inc_desc}. ETA: {eta} hours."
        )
    if not recommendations:
        recommendations.append(
            "Tower metrics are within normal parameters. Issue may be device-specific — recommend device restart."
        )

    return json.dumps({
        "tower_id": tower_id,
        "reported_symptom": symptom,
        "tower_status": status,
        "technology": tech,
        "diagnostics": {
            "avg_latency_ms": latency,
            "packet_loss_pct": pkt_loss,
            "throughput_mbps": throughput,
            "signal_strength_dbm": signal,
        },
        "recommendations": recommendations,
    })


async def get_regional_network_summary(region: str) -> str:
    """Return aggregated health summary for all towers in a region.

    Args:
        region: Region name (Midwest, Southwest, Northeast, West, Southeast).
    """
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT status, technology, COUNT(*) AS count
        FROM network_towers WHERE region = ?
        GROUP BY status, technology ORDER BY status
        """,
        (region,),
    )
    rows = cur.fetchall()
    cur.execute(
        """
        SELECT COUNT(*) FROM open_incidents i
        JOIN network_towers t ON i.tower_id = t.tower_id
        WHERE t.region = ?
        """,
        (region,),
    )
    incident_count = cur.fetchone()[0]
    conn.close()

    if not rows:
        return json.dumps({"error": f"No towers found for region: {region}"})

    return json.dumps({
        "region": region,
        "total_towers": sum(r[2] for r in rows),
        "breakdown": [{"status": r[0], "technology": r[1], "count": r[2]} for r in rows],
        "open_incidents": incident_count,
    })


from google.adk.agents import Agent

root_agent = Agent(
    name="network_diagnostics_agent",
    model=ADK_MODEL,
    description="Prodapt NOC specialist for tower and connectivity diagnostics.",
    instruction=(
        "You are a Network Operations Center (NOC) specialist for Prodapt Telecom. "
        "Diagnose tower connectivity issues, check tower status, and provide technical recommendations. "
        "Always use tools to query live SQL data — never guess. "
        "Identify the relevant tower ID or region, call the appropriate tool, "
        "and return a clear technical assessment with actionable recommendations. "
        "Reference specific metrics (latency, packet loss, signal strength) and incident IDs."
    ),
    tools=[check_tower_status, run_connectivity_diagnostics, get_regional_network_summary],
)

if __name__ == "__main__":
    import uvicorn
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    app = to_a2a(root_agent, host="localhost", port=PORT, protocol="http")
    print(f"Network Diagnostics ADK Agent starting on port {PORT}")
    print(f"Agent card: http://localhost:{PORT}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
