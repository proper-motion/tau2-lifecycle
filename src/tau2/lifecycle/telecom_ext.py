"""`telecom_lifecycle` domain: upstream telecom plus plan change, cancel and retention.

Upstream `telecom` is untouched (D13), so published τ²-bench scores stay
comparable. This domain adds three agent tools, a policy addendum, and
generated "intent" tasks for the calls a customer makes because the lifecycle
model decided they want to downgrade, upgrade or cancel.
See .plans/telecom-lifecycle-tools.md (D2-D5, OV1-OV9).
"""

from typing import Any, Dict, Literal

from tau2.data_model.tasks import Task
from tau2.domains.telecom.data_model import AccountStatus, LineStatus, TelecomDB
from tau2.domains.telecom.environment import TelecomEnvironment
from tau2.domains.telecom.tools import TelecomTools
from tau2.domains.telecom.user_data_model import TelecomUserDB
from tau2.domains.telecom.user_tools import TelecomUserTools
from tau2.domains.telecom.utils import (
    TELECOM_DB_PATH,
    TELECOM_MAIN_POLICY_PATH,
    TELECOM_TECH_SUPPORT_POLICY_MANUAL_PATH,
    TELECOM_USER_DB_PATH,
)
from tau2.environment.toolkit import ToolType, is_tool
from tau2.utils import load_file

DOMAIN = "telecom_lifecycle"
RETENTION_CREDIT = 20.0
ANNUAL_CREDIT_CAP = 50.0
PLAN_FOR_INTENT = {"downgrade": "P1001", "upgrade": "P1003"}

POLICY_ADDENDUM = f"""
## Cancel Line
If the user asks to cancel a line:
- Make sure you know which line the user wants to cancel and why.
- Check whether the user is eligible for a retention offer: the line is Active and the customer's goodwill credit used this year plus ${RETENTION_CREDIT:.0f} is at most ${ANNUAL_CREDIT_CAP:.0f}.
- If eligible, offer exactly one goodwill credit of ${RETENTION_CREDIT:.0f} to keep the line. If the user accepts, apply it and do not cancel the line.
- If the user declines the offer, or is not eligible, cancel the line.
- Never refuse or delay a cancellation the user still wants after the offer.

## Goodwill Credit
Goodwill credit is only used as the retention offer described in Cancel Line. The amount is always ${RETENTION_CREDIT:.0f}. A customer may receive at most ${ANNUAL_CREDIT_CAP:.0f} of goodwill credit per year, including credit already used.
"""


class LifecycleTelecomTools(TelecomTools):
    """Upstream telecom tools plus plan change, line cancellation and retention credit."""

    @is_tool(ToolType.WRITE)
    def change_plan(
        self, customer_id: str, line_id: str, new_plan_id: str
    ) -> Dict[str, Any]:
        """
        Changes the plan of a line.
        Checks: Line status must be Active; the new plan must exist and differ from the current plan.
        Logic: Sets the line's plan and records the plan change date.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to change.
            new_plan_id: ID of the plan to switch to.

        Returns:
            Dictionary with a message, the old and new monthly prices, and the updated line.

        Raises:
            ValueError: If customer, line or plan not found, line not active, or plan unchanged.
        """
        line = self._get_target_line(customer_id, line_id)
        if line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to change plan")
        if line.plan_id == new_plan_id:
            raise ValueError(f"Line {line_id} is already on plan {new_plan_id}")
        old_plan = self._get_plan_by_id(line.plan_id)
        new_plan = self._get_plan_by_id(new_plan_id)
        line.plan_id = new_plan.plan_id
        line.last_plan_change_date = self._get_today()
        return {
            "message": f"Plan changed from {old_plan.name} to {new_plan.name}.",
            "old_price_per_month": old_plan.price_per_month,
            "new_price_per_month": new_plan.price_per_month,
            "line": line,
        }

    @is_tool(ToolType.WRITE)
    def cancel_line(
        self, customer_id: str, line_id: str, reason: str
    ) -> Dict[str, Any]:
        """
        Cancels (closes) a line.
        Checks: Line must not already be Closed.
        Logic: Sets line status to Closed. If no other line of the customer remains open, closes the account.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to cancel.
            reason: Reason for cancellation.

        Returns:
            Dictionary with a message and the updated line.

        Raises:
            ValueError: If customer or line not found, or if the line is already closed.
        """
        line = self._get_target_line(customer_id, line_id)
        if line.status == LineStatus.CLOSED:
            raise ValueError(f"Line {line_id} is already closed")
        line.status = LineStatus.CLOSED
        customer = self.get_customer_by_id(customer_id)
        if all(
            self._get_line_by_id(lid).status == LineStatus.CLOSED
            for lid in customer.line_ids
        ):
            customer.account_status = AccountStatus.CLOSED
        return {"message": f"Line {line_id} cancelled.", "line": line}

    @is_tool(ToolType.WRITE)
    def apply_goodwill_credit(self, customer_id: str, amount: float) -> Dict[str, Any]:
        """
        Applies a goodwill credit to the customer's next bill.
        Checks: Amount must be positive; credit used this year plus amount must not exceed the annual cap.
        Logic: Adds the amount to the customer's goodwill credit used this year.

        Args:
            customer_id: ID of the customer.
            amount: Credit amount in USD.

        Returns:
            Dictionary with a message and the customer's credit used this year.

        Raises:
            ValueError: If customer not found, amount not positive, or annual cap exceeded.
        """
        if amount <= 0:
            raise ValueError("Credit amount must be positive")
        customer = self.get_customer_by_id(customer_id)
        if customer.goodwill_credit_used_this_year + amount > ANNUAL_CREDIT_CAP:
            raise ValueError(
                f"Annual goodwill credit cap of ${ANNUAL_CREDIT_CAP:.0f} would be exceeded"
            )
        customer.goodwill_credit_used_this_year += amount
        return {
            "message": f"Credit of ${amount:.2f} will appear on the next bill.",
            "goodwill_credit_used_this_year": customer.goodwill_credit_used_this_year,
        }


def get_environment(solo_mode: bool = False) -> TelecomEnvironment:
    """Upstream manual-policy telecom environment with the lifecycle tools and addendum."""
    if solo_mode:
        raise NotImplementedError("telecom_lifecycle supports conversational runs only")
    policy = (
        "<main_policy>\n"
        + load_file(TELECOM_MAIN_POLICY_PATH)
        + POLICY_ADDENDUM
        + "\n</main_policy>\n<tech_support_policy>\n"
        + load_file(TELECOM_TECH_SUPPORT_POLICY_MANUAL_PATH)
        + "\n</tech_support_policy>"
    )
    return TelecomEnvironment(
        domain_name=DOMAIN,
        policy=policy,
        tools=LifecycleTelecomTools(TelecomDB.load(TELECOM_DB_PATH)),
        user_tools=TelecomUserTools(TelecomUserDB.load(TELECOM_USER_DB_PATH)),
    )


# ---------------------------------------------------------------- intent tasks

Intent = Literal["downgrade", "upgrade", "cancel"]


def retention_eligible(state: dict) -> bool:
    """Same rule the policy gives the agent."""
    return (
        state.get("account_status", "Active") == "Active"
        and state.get("goodwill_credit_used_this_year", 25.0) + RETENTION_CREDIT
        <= ANNUAL_CREDIT_CAP
    )


def make_intent_task(state: dict, intent: Intent, accepts_offer: bool) -> Task:
    """A call the customer makes because the lifecycle model chose `intent`.

    `accepts_offer` comes from the customer's hidden price sensitivity (D5). The
    gold path follows the policy: an eligible cancel that accepts keeps the line
    with a credit; otherwise the line is cancelled. Returned task still uses the
    template customer and is personalised by `world.instantiate_task`.
    """
    from tau2.lifecycle.world import TEMPLATE_NAME

    phone = "555-123-2002"
    known = f"You are {TEMPLATE_NAME} with phone number {phone}."
    if intent == "cancel":
        offer = accepts_offer and retention_eligible(state)
        reason = "You want to cancel your phone line because the service is not worth what you pay."
        behaviour = (
            f"If the agent offers you a ${RETENTION_CREDIT:.0f} credit to stay, accept it and keep your line."
            if accepts_offer
            else "If the agent offers you anything to stay, politely refuse and insist on cancelling."
        )
        gold = (
            [
                (
                    "apply_goodwill_credit",
                    {"customer_id": "C1001", "amount": RETENTION_CREDIT},
                )
            ]
            if offer
            else [
                (
                    "cancel_line",
                    {
                        "customer_id": "C1001",
                        "line_id": "L1002",
                        "reason": "customer request",
                    },
                )
            ]
        )
    else:
        plan_id = PLAN_FOR_INTENT[intent]
        plan_name = {"P1001": "Basic Plan", "P1003": "Unlimited Plus"}[plan_id]
        reason = f"You want to switch your line to the {plan_name}."
        behaviour = "Confirm the new price when the agent asks."
        gold = [
            (
                "change_plan",
                {"customer_id": "C1001", "line_id": "L1002", "new_plan_id": plan_id},
            )
        ]
    variant = "accept" if accepts_offer else "refuse"
    return Task.model_validate(
        {
            "id": f"[lifecycle]{intent}_{variant}[PERSONA:{state['persona']}]",
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "domain": "telecom",
                    "reason_for_call": reason,
                    "known_info": known,
                    "unknown_info": None,
                    "task_instructions": behaviour,
                },
            },
            "initial_state": {
                "initialization_actions": [
                    {
                        "env_type": "user",
                        "func_name": "set_user_info",
                        "arguments": {"name": TEMPLATE_NAME, "phone_number": phone},
                    }
                ]
            },
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": f"{name}_0",
                        "requestor": "assistant",
                        "name": name,
                        "arguments": args,
                    }
                    for name, args in gold
                ],
                "reward_basis": ["DB"],
            },
        }
    )
