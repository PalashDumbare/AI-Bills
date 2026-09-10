from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(Chunk(
                text=chunk_text_str,
                index=chunk_index,
                start_char=start,
                end_char=end,
            ))
            chunk_index += 1

        start = end - overlap
        if start >= len(text):
            break

    return chunks
