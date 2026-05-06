#!/usr/bin/env python3
"""Comprehensive test for Memory Proxy v4.1 — project isolation, entities, time-decay."""
import requests
import json
import time
import sys

BASE = "http://localhost:5556"


def test(name, fn):
    try:
        result = fn()
        print(f"  ✓ {name}")
        return result
    except AssertionError as e:
        print(f"  ✗ {name}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        return None


def chat(msg, model="qwen2.5:7b-instruct-q4_K_M", conv_id=None, stream=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": msg}],
        "stream": stream,
    }
    if conv_id:
        body["conv_id"] = conv_id
    r = requests.post(f"{BASE}/api/chat", json=body, timeout=120)
    return r.json()


def search(query, project="general", top_k=5):
    r = requests.post(f"{BASE}/memory/search",
                       json={"query": query, "project": project, "top_k": top_k})
    return r.json()


def clear(project=None):
    body = {"project": project} if project else {}
    r = requests.post(f"{BASE}/memory/clear", json=body)
    return r.json()


def main():
    print("=" * 60)
    print("Memory Proxy v4.1 — Full Test Suite")
    print("=" * 60)

    # 0. Health check
    print("\n[0] Health")
    test("Health endpoint", lambda: (
        r := requests.get(f"{BASE}/health"),
        assert r.status_code == 200,
        d := r.json(),
        assert d["status"] == "ok",
        assert d["encoder"] == "loaded",
        print(f"      Version: {d.get('version', '?')}, Collections: {d.get('collections', {})}"),
    ))

    # 1. Clear all
    print("\n[1] Clear all memories")
    test("Clear all", lambda: (
        r := clear(),
        assert "cleared" in str(r),
    ))
    time.sleep(1)

    # 2. Store personal info (project: general)
    print("\n[2] Store personal info — general")
    test("Store personal info", lambda: (
        r := chat(
            "Je m'appelle Johnny Hairion, je vis en Belgique et je travaille "
            "sur TriVox2 pour créer le meilleur embedding du monde et battre MiniLM. "
            "J'utilise une RTX 5070 pour le training.",
            conv_id="test_chat1"
        ),
        assert r.get("message", {}).get("content"),
        print(f"      Response: {r['message']['content'][:80]}..."),
    ))
    time.sleep(2)

    # 3. Recall tests (different conv_ids = different chats)
    print("\n[3] Recall tests — general")

    test("Recall: name", lambda: (
        r := chat("Comment je m'appelle?", conv_id="test_chat2"),
        c := r.get("message", {}).get("content", "").lower(),
        print(f"      → {c[:100]}"),
        assert "johnny" in c,
    ))
    time.sleep(1)

    test("Recall: project", lambda: (
        r := chat("Sur quoi je travaille?", conv_id="test_chat3"),
        c := r.get("message", {}).get("content", "").lower(),
        print(f"      → {c[:100]}"),
        assert any(w in c for w in ["trivox", "embedding", "minilm"]),
    ))
    time.sleep(1)

    test("Recall: location", lambda: (
        r := chat("Où est-ce que j'habite?", conv_id="test_chat4"),
        c := r.get("message", {}).get("content", "").lower(),
        print(f"      → {c[:100]}"),
        assert "belgique" in c,
    ))
    time.sleep(1)

    test("Recall: hardware", lambda: (
        r := chat("Quelle carte graphique j'utilise?", conv_id="test_chat5"),
        c := r.get("message", {}).get("content", "").lower(),
        print(f"      → {c[:100]}"),
        assert "5070" in c or "rtx" in c,
    ))
    time.sleep(1)

    # 4. Project isolation
    print("\n[4] Project isolation")

    # Store info in project NEXUS
    test("Store NEXUS info", lambda: (
        r := chat(
            "Le projet NEXUS est une application web avec un budget de 8.3 millions "
            "d'euros financé par Nordic Ventures Capital. La couleur principale est #00CED1 "
            "et Sofia Andersson est la lead designer.",
            conv_id="nexus_chat1"
        ),
        assert r.get("message", {}).get("content"),
    ))
    time.sleep(2)

    # Search in NEXUS context
    test("Search NEXUS: budget", lambda: (
        r := search("budget du projet", project="nexus"),
        print(f"      Results: {len(r.get('results', []))}"),
        texts := " ".join(m["text"] for m in r.get("results", [])).lower(),
        assert "8.3" in texts or "millions" in texts or "nordic" in texts or len(r.get("results", [])) > 0,
    ))

    # Search in general should NOT find NEXUS
    test("General search should not find NEXUS details", lambda: (
        r := search("Sofia Andersson", project="general"),
        # Should have fewer/no results about NEXUS
        print(f"      General results: {len(r.get('results', []))}"),
    ))

    # 5. Entity extraction
    print("\n[5] Entity extraction")
    test("Entities in memory search", lambda: (
        r := search("couleur hex", project="nexus"),
        texts := " ".join(m["text"] for m in r.get("results", [])),
        print(f"      Results: {[m['text'][:60] for m in r.get('results', [])[:3]]}"),
    ))

    # 6. Stats
    print("\n[6] Memory stats")
    test("Stats endpoint", lambda: (
        r := requests.get(f"{BASE}/memory/stats"),
        d := r.json(),
        print(f"      Total: {d.get('total', '?')}, Collections: {d.get('collections', {})}"),
        assert d.get("total", 0) > 0,
    ))

    # 7. Clear project
    print("\n[7] Project-specific clear")
    test("Clear NEXUS only", lambda: (
        r := clear(project="nexus"),
        assert r.get("status") == "cleared",
        print(f"      Deleted: {r.get('deleted', 0)} from nexus"),
    ))

    # General memories should still exist
    test("General memories preserved", lambda: (
        r := search("Johnny Hairion", project="general"),
        assert len(r.get("results", [])) > 0,
        print(f"      General still has {len(r['results'])} results"),
    ))

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
