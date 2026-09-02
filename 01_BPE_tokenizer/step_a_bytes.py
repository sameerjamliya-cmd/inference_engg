text = "hello 👋 world"

byte_string = text.encode("utf-8")     # str -> bytes (UTF-8 encoding)
token_ids = list(byte_string)          # bytes -> list[int], each in [0, 255]

print("text:      ", text)
print("byte_string:", byte_string)
print("token_ids: ", token_ids)
print("num tokens:", len(token_ids))
