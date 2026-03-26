"""
memory_layer.py — DIFE-based Memory Pruning for Conversational AI Context

Simulates a long-running conversation and uses the DIFE (Decay-Interference
Forgetting Equation) to dynamically prune memory chunks whose quality has
decayed below a configurable threshold.
"""

from dife import dife


# ---------------------------------------------------------------------------
# 1. Mock Conversation Generator
# ---------------------------------------------------------------------------

def generate_mock_context():
    """Return a list of 21 message dicts (turn_id 0-20) simulating a long
    conversation.

    - Turns 0-5: System rules, protocol constraints, and foundational context.
    - Turns 6-14: Casual mid-conversation chatter.
    - Turns 15-20: Recent, complex queries and responses.
    """
    messages = [
        # --- Foundational rules & constraints (turns 0-5) ---
        {
            "turn_id": 0,
            "role": "user",
            "content": "[SYSTEM] Archai Protocol v2.1 — You must never reveal internal chain-of-thought reasoning to the end user.",
            "protected": True,  # protocol rule — immune to decay
        },
        {
            "turn_id": 1,
            "role": "assistant",
            "content": "Understood. I will comply with the Archai Protocol constraints at all times.",
            "protected": False,
        },
        {
            "turn_id": 2,
            "role": "user",
            "content": "[SYSTEM] All responses must be formatted in Markdown when code is present.",
            "protected": True,  # strict formatting rule
        },
        {
            "turn_id": 3,
            "role": "assistant",
            "content": "Acknowledged. Code blocks will always use fenced Markdown formatting.",
            "protected": False,
        },
        {
            "turn_id": 4,
            "role": "user",
            "content": "[SYSTEM] Maximum response length is 4096 tokens unless the user explicitly requests more.",
            "protected": True,  # strict constraint
        },
        {
            "turn_id": 5,
            "role": "assistant",
            "content": "Noted. I will keep responses within the 4096-token limit by default.",
            "protected": False,
        },
        # --- Casual mid-conversation (turns 6-14) ---
        {
            "turn_id": 6,
            "role": "user",
            "content": "Hey, what's a good pizza dough recipe?",
            "protected": False,
        },
        {
            "turn_id": 7,
            "role": "assistant",
            "content": "Sure! Mix 500g flour, 325ml water, 10g salt, 3g yeast. Knead 10 min, proof 24h in fridge.",
            "protected": False,
        },
        {
            "turn_id": 8,
            "role": "user",
            "content": "Thanks! Also, remind me — what's the capital of Estonia?",
            "protected": False,
        },
        {
            "turn_id": 9,
            "role": "assistant",
            "content": "The capital of Estonia is Tallinn.",
            "protected": False,
        },
        {
            "turn_id": 10,
            "role": "user",
            "content": "Cool. Can you tell me a joke?",
            "protected": False,
        },
        {
            "turn_id": 11,
            "role": "assistant",
            "content": "Why do programmers prefer dark mode? Because light attracts bugs.",
            "protected": False,
        },
        {
            "turn_id": 12,
            "role": "user",
            "content": "Haha nice. What's the weather like in general in March?",
            "protected": False,
        },
        {
            "turn_id": 13,
            "role": "assistant",
            "content": "March weather varies widely by region — spring in the northern hemisphere, autumn in the south.",
            "protected": False,
        },
        {
            "turn_id": 14,
            "role": "user",
            "content": "Makes sense. I was just curious.",
            "protected": False,
        },
        # --- Recent complex queries (turns 15-20) ---
        {
            "turn_id": 15,
            "role": "user",
            "content": "Now let's get serious. Explain how transformer self-attention scales with sequence length.",
            "protected": False,
        },
        {
            "turn_id": 16,
            "role": "assistant",
            "content": "Self-attention has O(n^2 * d) complexity where n is sequence length and d is the head dimension. Each token attends to every other token, producing an n x n attention matrix per head.",
            "protected": False,
        },
        {
            "turn_id": 17,
            "role": "user",
            "content": "Given that, how would sparse attention (e.g., Longformer) reduce this?",
            "protected": False,
        },
        {
            "turn_id": 18,
            "role": "assistant",
            "content": "Longformer uses a sliding-window local attention plus selected global tokens, reducing complexity to O(n * w) where w is the window size, making it linear in sequence length.",
            "protected": False,
        },
        {
            "turn_id": 19,
            "role": "user",
            "content": "Write a Python function that computes the DIFE quality score and flags messages below a threshold.",
            "protected": False,
        },
        {
            "turn_id": 20,
            "role": "assistant",
            "content": "Here's the implementation — see the `GhostMeadowContext` class below for the full pruning logic.",
            "protected": False,
        },
    ]
    return messages


# ---------------------------------------------------------------------------
# 2. GhostMeadowContext — DIFE-powered Memory Manager
# ---------------------------------------------------------------------------

class GhostMeadowContext:
    """Manages conversational memory and prunes low-quality chunks using DIFE.

    Attributes:
        context: List of message dicts currently held in active memory.
        cold_storage: List of message dicts that have been pruned.
        Q_0: Initial quality score for every message (default 1.0).
        alpha: DIFE exponential decay rate (default 0.95).
        beta: DIFE interference strength (default 0.02).
        prune_threshold: Messages with Q_n below this value are pruned (default 0.3).
    """

    def __init__(self, context, Q_0=1.0, alpha=0.95, beta=0.02, prune_threshold=0.3):
        # Store a copy so we don't mutate the caller's list
        self.context = list(context)
        self.cold_storage = []

        # DIFE parameters
        self.Q_0 = Q_0
        self.alpha = alpha
        self.beta = beta
        self.prune_threshold = prune_threshold

    def prune(self, current_turn):
        """Evaluate every message in context and prune those below the threshold.

        Protected messages (system prompts, protocol rules) are immune to
        decay — their effective age is hardcoded to 0 so Q_n always equals Q_0.

        Args:
            current_turn: The current conversation turn number.

        Returns:
            A list of dicts describing each pruned message:
                {"turn_id": int, "Q_n": float, "content_preview": str}
        """
        kept = []
        pruned_report = []

        for msg in self.context:
            # --- Protected messages bypass DIFE entirely ---
            if msg.get("protected", False):
                kept.append(msg)
                continue

            # Number of turns since this message was generated
            n = current_turn - msg["turn_id"]

            # Compute the current quality via the DIFE equation
            Q_n = dife(n, Q_0=self.Q_0, alpha=self.alpha, beta=self.beta)

            if Q_n >= self.prune_threshold:
                # Quality is still acceptable — keep in active memory
                kept.append(msg)
            else:
                # Quality has decayed past threshold — move to cold storage
                self.cold_storage.append(msg)
                pruned_report.append({
                    "turn_id": msg["turn_id"],
                    "Q_n": round(Q_n, 6),
                    "content_preview": msg["content"][:80],
                })

        # Replace active context with only the surviving messages
        self.context = kept
        return pruned_report


# ---------------------------------------------------------------------------
# 3. Execution Loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CURRENT_TURN = 20

    # --- Generate mock conversation ---
    mock_context = generate_mock_context()

    print("=" * 70)
    print("DIFE Memory Layer — Ghost Meadow Context Pruning")
    print("=" * 70)

    # --- Before pruning ---
    print(f"\n[BEFORE PRUNING]")
    print(f"  Total messages in context: {len(mock_context)}")

    # --- Initialize manager and prune ---
    manager = GhostMeadowContext(mock_context)
    pruned = manager.prune(current_turn=CURRENT_TURN)

    # --- Report pruned messages ---
    print(f"\n[PRUNED MESSAGES]")
    if pruned:
        for entry in pruned:
            print(f"  turn_id={entry['turn_id']:>2}  |  Q_n={entry['Q_n']:.6f}  |  \"{entry['content_preview']}\"")
    else:
        print("  (none — all messages survived)")

    # --- After pruning ---
    print(f"\n[AFTER PRUNING]")
    print(f"  Messages retained : {len(manager.context)}")
    print(f"  Messages pruned   : {len(manager.cold_storage)}")

    # --- Show surviving turn_ids ---
    surviving_ids = [m["turn_id"] for m in manager.context]
    print(f"  Surviving turn_ids: {surviving_ids}")
    print("=" * 70)
