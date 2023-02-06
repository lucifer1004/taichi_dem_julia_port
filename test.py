Integer = int


def morton3d32(x: Integer, y: Integer, z: Integer) -> Integer:
    answer = 0
    x &= 0x3ff
    x = (x | x << 16) & 0x30000ff
    x = (x | x << 8) & 0x300f00f
    x = (x | x << 4) & 0x30c30c3
    x = (x | x << 2) & 0x9249249
    y &= 0x3ff
    y = (y | y << 16) & 0x30000ff
    y = (y | y << 8) & 0x300f00f
    y = (y | y << 4) & 0x30c30c3
    y = (y | y << 2) & 0x9249249
    z &= 0x3ff
    z = (z | z << 16) & 0x30000ff
    z = (z | z << 8) & 0x300f00f
    z = (z | z << 4) & 0x30c30c3
    z = (z | z << 2) & 0x9249249
    answer |= x | y << 1 | z << 2
    return answer


print(morton3d32(19, 2, 3))
