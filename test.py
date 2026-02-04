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
            with self.subTest(video=video.format.name):
                mini_BM3D(video, sigma=5)
                mini_BM3D(video, sigma=5, planes=[0])
            with self.subTest(video=video.format.name):
                mini_BM3D(video, sigma=5, ref=video)
                mini_BM3D(video, sigma=5, ref=video, planes=[0])
            with self.subTest(video=video.format.name):
                mini_BM3D(video, sigma=5, accel="CPU")
                mini_BM3D(video, sigma=5, planes=[0], accel="CPU")
            with self.subTest(video=video.format.name):
                mini_BM3D(video, sigma=5, ref=video, accel="CPU")
                mini_BM3D(video, sigma=5, ref=video, planes=[0], accel="CPU")

    def test_adenoise(self):
        """ Test adenoise with all defaults """
        from addfunc.adfunc import adenoise

        videos = self.videos[0:4]  # RGB not supported
        for video in videos:
            with self.subTest(video=video.format.name):
                adenoise.scan8mm(video)
            with self.subTest(video=video.format.name):
                adenoise.scan16mm(video)
            with self.subTest(video=video.format.name):
                adenoise.scan35mm(video)
            with self.subTest(video=video.format.name):
                adenoise.scan65mm(video)
            with self.subTest(video=video.format.name):
                adenoise.digital(video)
            with self.subTest(video=video.format.name):
                adenoise.default(video)
                adenoise.default(video, chroma_denoise="cbm3d")
                adenoise.default(video, chroma_denoise="artcnn")

    def test_auto_deblock(self):
        """ Test auto_deblock """
        from addfunc.adfunc import auto_deblock

        videos = self.videos[1:4]  # Grayscale and RGB not supported
        for video in videos:
            with self.subTest(video=video.format.name):
                auto_deblock(video)
            with self.subTest(video=video.format.name):
                auto_deblock(video, pre=True)
            with self.subTest(video=video.format.name):
                auto_deblock(video, planes=[0])
                auto_deblock(video, planes=[1])
            with self.subTest(video=video.format.name):
                auto_deblock(video, pre=True, planes=[0])

    def test_msaa2x(self):
        """ Test msaa2x """
        from addfunc.adfunc import msaa2x

        videos = self.videos[0:4]
        for video in videos:
            with self.subTest(video=video.format.name):
                msaa2x(video)
            with self.subTest(video=video.format.name):
                msaa2x(video, ref=video)
            with self.subTest(video=video.format.name):
                msaa2x(video, planes=[0])
            with self.subTest(video=video.format.name):
                msaa2x(video, ref=video, planes=[0])

    def test_luma_masks(self):
        """ Test luma-based masks """
        from addfunc.admask import luma_mask, luma_mask_man, luma_mask_ping

        for video in self.videos:
            with self.subTest(video=video.format.name):
                luma_mask(video)
            with self.subTest(video=video.format.name):
                luma_mask_man(video)
            with self.subTest(video=video.format.name):
                luma_mask_ping(video)

    def test_edgemasks(self):
        """ Test edgemasks """
        from addfunc.admask import advanced_edgemask

        videos = self.videos[0:4]
        for video in videos:
            with self.subTest(video=video.format.name):
                advanced_edgemask(video)

    def test_retinex(self):
        """ Test retinex """
        from addfunc.admask import unbloat_retinex

        videos = self.videos[0]
        for video in videos:
            with self.subTest(video=video.format.name):
                unbloat_retinex(video)
        
    def test_hd_flatmask(self):
        """ Test hd_flatmask """
        from addfunc.admask import hd_flatmask

        videos = self.videos[0:4]
        for video in videos:
            with self.subTest(video=video.format.name):
                hd_flatmask(video)

if __name__ == '__main__':
    unittest.main()