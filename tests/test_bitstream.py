from ambi.io import BitWriter, BitReader

def test_varint_roundtrip():
    bw = BitWriter()
    for v in [0,1,2,127,128,255,256,16384,1<<31]:
        bw.write_varint(v)
    br = BitReader(bw.getvalue())
    vals = []
    for _ in range(9):
        vals.append(br.read_varint())
    assert vals == [0,1,2,127,128,255,256,16384,1<<31]