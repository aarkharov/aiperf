# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.models import Conversation, Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
    MemoryMapDatasetClientStore,
)


def _make_conversation(
    session_id: str, raw_payloads: list[dict | None]
) -> Conversation:
    """Create a conversation with optional raw_payload turns."""
    turns = []
    for payload in raw_payloads:
        if payload is not None:
            turns.append(Turn(role="user", raw_payload=payload))
        else:
            turns.append(Turn(role="user"))
    return Conversation(session_id=session_id, turns=turns)


@pytest.mark.asyncio
async def test_payload_mmap_round_trip(tmp_path, monkeypatch):
    """Test writing and reading payload data through the mmap backing store."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(benchmark_id="test_payload")
    await store.initialize()

    payload_1 = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}
    payload_2 = {"messages": [{"role": "user", "content": "World"}], "model": "gpt-4"}

    conv1 = _make_conversation("conv-1", [payload_1, payload_2])
    conv2 = _make_conversation("conv-2", [None])

    await store.add_conversation("conv-1", conv1)
    await store.add_conversation("conv-2", conv2)
    await store.finalize()

    metadata = store.get_client_metadata()
    assert metadata.payload_data_file_path is not None
    assert metadata.payload_index_file_path is not None

    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
        payload_data_file_path=metadata.payload_data_file_path,
        payload_index_file_path=metadata.payload_index_file_path,
    )

    # Check payload bytes for conv-1
    pb0 = client.get_payload_bytes("conv-1", 0)
    assert pb0 is not None
    assert orjson.loads(pb0) == payload_1

    pb1 = client.get_payload_bytes("conv-1", 1)
    assert pb1 is not None
    assert orjson.loads(pb1) == payload_2

    # conv-2 has no raw_payload
    assert client.get_payload_bytes("conv-2", 0) is None

    # Out of range
    assert client.get_payload_bytes("conv-1", 99) is None

    # Non-existent conversation
    assert client.get_payload_bytes("conv-999", 0) is None

    client.close()
    await store.stop()


@pytest.mark.asyncio
async def test_no_payload_data_no_files(tmp_path, monkeypatch):
    """When no conversations have raw_payload, payload files should not be created."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(benchmark_id="test_no_payload")
    await store.initialize()

    conv = _make_conversation("conv-1", [None])
    await store.add_conversation("conv-1", conv)
    await store.finalize()

    metadata = store.get_client_metadata()
    assert metadata.payload_data_file_path is None
    assert metadata.payload_index_file_path is None

    await store.stop()


@pytest.mark.asyncio
async def test_client_store_get_payload_bytes(tmp_path, monkeypatch):
    """Test MemoryMapDatasetClientStore.get_payload_bytes async wrapper."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(benchmark_id="test_client_payload")
    await store.initialize()

    payload = {"messages": [{"role": "user", "content": "test"}]}
    conv = _make_conversation("conv-1", [payload])
    await store.add_conversation("conv-1", conv)
    await store.finalize()

    metadata = store.get_client_metadata()
    client_store = MemoryMapDatasetClientStore(client_metadata=metadata)
    await client_store.initialize()

    result = await client_store.get_payload_bytes("conv-1", 0)
    assert result is not None
    assert orjson.loads(result) == payload

    result_none = await client_store.get_payload_bytes("conv-1", 99)
    assert result_none is None

    await client_store.stop()
    await store.stop()
