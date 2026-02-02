import unittest
from vstools import vs

class Test(unittest.TestCase):
    def setUp(self):
        c = vs.core
        self.video400 = c.std.BlankClip(width=1280, height=720, format=vs.GRAY8, length=100, color=64)
        self.video420 = c.std.BlankClip(width=1280, height=720, format=vs.YUV420P8, length=100, color=[64, 64, 64])
        self.video422 = c.std.BlankClip(width=1280, height=720, format=vs.YUV422P8, length=100, color=[64, 64, 64])
        self.video444 = c.std.BlankClip(width=1280, height=720, format=vs.YUV444P8, length=100, color=[64, 64, 64])
        self.videoRGB = c.std.BlankClip(width=1280, height=720, format=vs.RGB24, length=100, color=[64, 64, 64])
        self.videos = [self.video400, self.video420, self.video422, self.video444, self.videoRGB]
        self.videos = [v.noise.Add(5, 2, type=3, xsize=4, ysize=4, constant=False) for v in self.videos]    # add dynamic noise

    def test_mini_BM3D(self):
        """ Test mini_BM3D with and without reference clip """
        from addfunc.adfunc import mini_BM3D

        for video in self.videos:
            with self.subTest(video=video):
                mini_BM3D(video, sigma=5)
                mini_BM3D(video, sigma=5, planes=[0])
            with self.subTest(video=video):
                mini_BM3D(video, sigma=5, ref=video)
                mini_BM3D(video, sigma=5, ref=video, planes=[0])
            with self.subTest(video=video):
                mini_BM3D(video, sigma=5, accel="CPU")
                mini_BM3D(video, sigma=5, planes=[0], accel="CPU")
            with self.subTest(video=video):
                mini_BM3D(video, sigma=5, ref=video, accel="CPU")
                mini_BM3D(video, sigma=5, ref=video, planes=[0], accel="CPU")

    def test_adenoise(self):
        """ Test adenoise with all defaults and with reference clip """
        from addfunc.adfunc import adenoise

        for video in self.videos:
            with self.subTest(video=video):
                adenoise.scan8mm(video)
            with self.subTest(video=video):
                adenoise.scan16mm(video)
            with self.subTest(video=video):
                adenoise.scan35mm(video)
            with self.subTest(video=video):
                adenoise.digital(video)
            with self.subTest(video=video):
                adenoise.default(video)

if __name__ == '__main__':
    unittest.main()