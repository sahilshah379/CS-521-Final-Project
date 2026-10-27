import re

CONFIRM = "user_confirmation"
AUTH = ("find_user_id_by_email", "find_user_id_by_name_zip")
YES = re.compile(r"^\W*(yes|yeah|yep|sure|confirm|correct|go ahead|please do|ok|okay)\b", re.I)

# tool a -> prerequisites p in P_a, each (alternatives, bound argument); a rule is ¬(¬p U a)
_WRITE_AIRLINE = [(("get_reservation_details",), "reservation_id"), ((CONFIRM,), None)]
_WRITE_RETAIL = [(AUTH, None), (("get_order_details",), "order_id"), ((CONFIRM,), None)]

RULES = {
    "airline": {
        "book_reservation": [(("get_user_details",), "user_id"), ((CONFIRM,), None)],
        "cancel_reservation": _WRITE_AIRLINE,
        "update_reservation_flights": _WRITE_AIRLINE,
        "update_reservation_baggages": _WRITE_AIRLINE,
        "update_reservation_passengers": _WRITE_AIRLINE,
        "send_certificate": [(("get_user_details",), "user_id")],
    },
    "retail": {
        "get_user_details": [(AUTH, None)],
        "get_order_details": [(AUTH, None)],
        "modify_user_address": [(AUTH, None), ((CONFIRM,), None)],
        "cancel_pending_order": _WRITE_RETAIL,
        "modify_pending_order_address": _WRITE_RETAIL,
        "modify_pending_order_items": _WRITE_RETAIL,
        "modify_pending_order_payment": _WRITE_RETAIL,
        "return_delivered_order_items": _WRITE_RETAIL,
        "exchange_delivered_order_items": _WRITE_RETAIL,
    },
}


class Checker:
    def __init__(self, domain):
        self.rules, self.events = RULES[domain], []

    def observe_user(self, text):
        if text and YES.match(text):
            self.events.append((CONFIRM, {}))

    def record(self, calls):
        self.events += [(c.name, c.arguments) for c in calls]

    def check(self, calls):
        """Returns the first rule the proposed calls would violate, or None."""
        events = list(self.events)
        for c in calls:
            for alts, key in self.rules.get(c.name, []):
                if not any(n in alts and (key is None or args.get(key) == c.arguments.get(key)) for n, args in events):
                    return f"{c.name} requires {' or '.join(alts)}" + (f" on the same {key}" if key else "") + " first"
            events.append((c.name, c.arguments))
        return None
