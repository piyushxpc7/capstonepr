"""Billing Resolution ADK Agent — exposed via A2A on port 8002.

Start with: python adk-services/billing_resolution/agent.py
Agent card will be at: http://localhost:8002/.well-known/agent-card.json
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

DB_PATH = os.environ.get(
    "DB_PATH",
    str(Path(__file__).parent.parent.parent / "data" / "telecom_ops.db"),
)
ADK_MODEL = os.environ.get("ADK_MODEL", "gemini-2.0-flash-exp")
PORT = 8002
AUTO_CREDIT_LIMIT = 50.0


def _db():
    return sqlite3.connect(DB_PATH)


async def lookup_billing_account(customer_id: str) -> str:
    """Return account balance, plan details, and recent charges for a customer.

    Args:
        customer_id: The customer identifier (e.g. CUST-10002).
    """
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.customer_id, a.current_balance, a.billing_cycle_day,
               s.plan_name, s.account_type, s.monthly_fee
        FROM billing_accounts a
        JOIN customer_subscriptions s ON a.customer_id = s.customer_id
        WHERE a.customer_id = ?
        """,
        (customer_id,),
    )
    account = cur.fetchone()

    if not account:
        conn.close()
        return json.dumps({"error": f"Customer {customer_id} not found."})

    cur.execute(
        """
        SELECT charge_id, description, amount, charge_date, billing_period, is_duplicate_flag
        FROM billing_charges
        WHERE customer_id = ?
        ORDER BY charge_date DESC LIMIT 10
        """,
        (customer_id,),
    )
    charges = cur.fetchall()
    conn.close()

    return json.dumps({
        "customer_id": account[0],
        "current_balance": account[1],
        "billing_cycle_day": account[2],
        "plan_name": account[3],
        "account_type": account[4],
        "monthly_fee": account[5],
        "recent_charges": [
            {
                "charge_id": c[0],
                "description": c[1],
                "amount": c[2],
                "charge_date": c[3],
                "billing_period": c[4],
                "is_duplicate": bool(c[5]),
            }
            for c in charges
        ],
    })


async def check_duplicate_charges(customer_id: str) -> str:
    """Find duplicate or flagged charges on a customer account.

    Args:
        customer_id: The customer identifier (e.g. CUST-10002).
    """
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT charge_id, description, amount, charge_date, billing_period
        FROM billing_charges
        WHERE customer_id = ? AND is_duplicate_flag = 1
        ORDER BY charge_date DESC
        """,
        (customer_id,),
    )
    duplicates = cur.fetchall()
    conn.close()

    if not duplicates:
        return json.dumps({
            "customer_id": customer_id,
            "duplicate_charges_found": False,
            "message": "No duplicate charges flagged on this account.",
        })

    return json.dumps({
        "customer_id": customer_id,
        "duplicate_charges_found": True,
        "duplicates": [
            {
                "charge_id": d[0],
                "description": d[1],
                "amount": d[2],
                "charge_date": d[3],
                "billing_period": d[4],
            }
            for d in duplicates
        ],
    })


async def apply_billing_credit(customer_id: str, amount: float, reason: str) -> str:
    """Apply a billing credit to a customer account.

    Credits up to $50 are auto-approved (APPLIED). Over $50 requires supervisor approval (PENDING_APPROVAL).

    Args:
        customer_id: The customer identifier (e.g. CUST-10002).
        amount: Credit amount in USD (positive number).
        reason: Reason for the credit (e.g. 'Duplicate charge refund').
    """
    status = "APPLIED" if amount <= AUTO_CREDIT_LIMIT else "PENDING_APPROVAL"
    reference = f"CR-{uuid.uuid4().hex[:8].upper()}"
    applied_at = datetime.now(timezone.utc).isoformat()

    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO billing_credits (customer_id, amount, reason, status, reference_number, applied_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (customer_id, amount, reason, status, reference, applied_at),
    )
    credit_id = cur.lastrowid

    new_balance = None
    if status == "APPLIED":
        cur.execute(
            "UPDATE billing_accounts SET current_balance = current_balance - ? WHERE customer_id = ?",
            (amount, customer_id),
        )
        cur.execute(
            "SELECT current_balance FROM billing_accounts WHERE customer_id = ?",
            (customer_id,),
        )
        row = cur.fetchone()
        new_balance = round(row[0], 2) if row else None

    conn.commit()
    conn.close()

    result = {
        "credit_id": credit_id,
        "customer_id": customer_id,
        "amount": amount,
        "reason": reason,
        "status": status,
        "reference_number": reference,
        "applied_at": applied_at,
    }
    if status == "APPLIED":
        result["new_balance"] = new_balance
        result["message"] = (
            f"Credit of ${amount:.2f} applied. New balance: ${new_balance:.2f}."
        )
    else:
        result["message"] = (
            f"Credit of ${amount:.2f} exceeds ${AUTO_CREDIT_LIMIT:.0f} auto-approval limit. "
            f"Submitted for supervisor approval (ref: {reference})."
        )

    return json.dumps(result)


from google.adk.agents import Agent

root_agent = Agent(
    name="billing_resolution_agent",
    model=ADK_MODEL,
    description="Prodapt billing specialist for investigating and resolving billing disputes.",
    instruction=(
        "You are a Billing Resolution specialist for Prodapt Telecom. "
        "Investigate billing disputes, identify duplicate charges, and apply credits "
        "following Prodapt's billing disputes policy. "
        "Always use tools to query live SQL data. "
        "For every dispute: look up the account, check for duplicates, determine the credit amount, "
        "and apply it using apply_billing_credit. "
        "Credits up to $50 are auto-approved; over $50 go to PENDING_APPROVAL. "
        "Always report the credit reference number, new balance, and approval status."
    ),
    tools=[lookup_billing_account, check_duplicate_charges, apply_billing_credit],
)

if __name__ == "__main__":
    import uvicorn
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    app = to_a2a(root_agent, host="localhost", port=PORT, protocol="http")
    print(f"Billing Resolution ADK Agent starting on port {PORT}")
    print(f"Agent card: http://localhost:{PORT}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
