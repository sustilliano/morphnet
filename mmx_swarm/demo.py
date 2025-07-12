from .memory_core import ImmediateMemory, SessionStore


def main():
    mem = ImmediateMemory()
    store = SessionStore(db_path=":memory:", keywords=["AI"])

    for i in range(1000):
        text = f"AI message {i}" if i % 100 == 0 else f"message {i}"
        store.insert("user", text)
        mem.append("user", text)

    matches = store.search("AI")
    print(f"Found {len(matches)} AI messages")


if __name__ == "__main__":
    main()
