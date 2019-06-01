import unittest

from io import BytesIO

from bdflib import model, reader

# This comes from the X11 BDF spec.
SAMPLE_FONT = b"""
STARTFONT 2.1
COMMENT This is a sample font in 2.1 format.
FONT -Adobe-Helvetica-Bold-R-Normal--24-240-75-75-P-65-ISO8859-1
SIZE 24 75 75
FONTBOUNDINGBOX 9 24 -2 -6
STARTPROPERTIES 19
FOUNDRY "Adobe"
FAMILY "Helvetica"
WEIGHT_NAME "Bold"
SLANT "R"
SETWIDTH_NAME "Normal"
ADD_STYLE_NAME ""
PIXEL_SIZE 24
POINT_SIZE 240
RESOLUTION_X 75
RESOLUTION_Y 75
SPACING "P"
AVERAGE_WIDTH 65
CHARSET_REGISTRY "ISO8859"
CHARSET_ENCODING "1"
MIN_SPACE 4
FONT_ASCENT 21
FONT_DESCENT 7
COPYRIGHT "Copyright (c) 1987 Adobe Systems, Inc."
NOTICE "Helvetica is a registered trademark of Linotype Inc."
ENDPROPERTIES
CHARS 2
STARTCHAR j
ENCODING 106
SWIDTH 355 0
DWIDTH 8 0
BBX 9 22 -2 -6
BITMAP
0380
0380
0380
0380
0000
0700
0700
0700
0700
0E00
0E00
0E00
0E00
0E00
1C00
1C00
1C00
1C00
3C00
7800
F000
E000
ENDCHAR
STARTCHAR quoteright
ENCODING 39
SWIDTH 223 0
DWIDTH 5 0
BBX 4 6 2 12
ATTRIBUTES 01C0
BITMAP
70
70
70
60
E0
C0
ENDCHAR
ENDFONT
""".strip()


class TestGlyph(unittest.TestCase):

	def test_basic_operation(self):
		testFont = model.Font(b"Adobe Helvetica", 24, 75, 75)
		testComments = []
		testGlyphData = reader._next_token(SAMPLE_FONT.split(b'\n')[27:56], testComments)

		reader._read_glyph(testGlyphData, testFont)

		# The font should now have an entry for j.
		testGlyph = testFont[106]

		# The glyph should have the correct header data.
		self.failUnlessEqual(testGlyph.name, b'j')
		self.failUnlessEqual(testGlyph.codepoint, 106)
		self.failUnlessEqual(testGlyph.advance, 8)
		self.failUnlessEqual(testGlyph.bbX, -2)
		self.failUnlessEqual(testGlyph.bbY, -6)
		self.failUnlessEqual(testGlyph.bbW,  9)
		self.failUnlessEqual(testGlyph.bbH, 22)

		# Make sure we got the correct glyph bitmap.
		self.failUnlessEqual(str(testGlyph),
				'..|...###\n'
				'..|...###\n'
				'..|...###\n'
				'..|...###\n'
				'..|......\n'
				'..|..###.\n'
				'..|..###.\n'
				'..|..###.\n'
				'..|..###.\n'
				'..|.###..\n'
				'..|.###..\n'
				'..|.###..\n'
				'..|.###..\n'
				'..|.###..\n'
				'..|###...\n'
				'--+###---\n'
				'..|###...\n'
				'..|###...\n'
				'..####...\n'
				'.####....\n'
				'####.....\n'
				'###......'
			)

		# The iterator should have nothing left in it.
		self.failUnlessRaises(StopIteration, next, testGlyphData)


class TestReadProperty(unittest.TestCase):

	def test_basic_operation(self):
		testFont = model.Font(b"Adobe Helvetica", 24, 75, 75)
		testComments = []
		testProperties = reader._next_token(SAMPLE_FONT.split(b'\n')[6:26], testComments)

		for i in range(19):
			key, value = next(testProperties)
			testFont[key] = reader._unquote_property_value(value)

		# After reading the properties, the iterator should be just up to the
		# ENDPROPERTIES line.
		self.failUnlessEqual(next(testProperties)[0], b"ENDPROPERTIES")

		# Test that the properties were read correctly.
		self.failUnlessEqual(testFont[b'FOUNDRY'], b"Adobe")
		self.failUnlessEqual(testFont[b'FAMILY'], b"Helvetica")
		self.failUnlessEqual(testFont[b'WEIGHT_NAME'], b"Bold")
		self.failUnlessEqual(testFont[b'SLANT'], b"R")
		self.failUnlessEqual(testFont[b'SETWIDTH_NAME'], b"Normal")
		self.failUnlessEqual(testFont[b'ADD_STYLE_NAME'], b"")
		self.failUnlessEqual(testFont[b'POINT_SIZE'], 24.0)
		self.failUnlessEqual(testFont[b'RESOLUTION_X'], 75)
		self.failUnlessEqual(testFont[b'RESOLUTION_Y'], 75)
		self.failUnlessEqual(testFont[b'SPACING'], b"P")
		self.failUnlessEqual(testFont[b'AVERAGE_WIDTH'], 65)
		self.failUnlessEqual(testFont[b'CHARSET_REGISTRY'], b"ISO8859")
		self.failUnlessEqual(testFont[b'CHARSET_ENCODING'], b"1")
		self.failUnlessEqual(testFont[b'MIN_SPACE'], 4)
		self.failUnlessEqual(testFont[b'FONT_ASCENT'], 21)
		self.failUnlessEqual(testFont[b'FONT_DESCENT'], 7)
		self.failUnlessEqual(testFont[b'COPYRIGHT'],
				b"Copyright (c) 1987 Adobe Systems, Inc.")
		self.failUnlessEqual(testFont[b'NOTICE'],
				b"Helvetica is a registered trademark of Linotype Inc.")
		self.failUnlessEqual(testComments, [])


class TestReadFont(unittest.TestCase):

	def _check_font(self, font):
		"""
		Checks that the given font is a representation of the sample font.
		"""
		self.failUnlessEqual(font[b"FACE_NAME"],
				b"-Adobe-Helvetica-Bold-R-Normal--24-240-75-75-P-65-ISO8859-1")
		self.failUnlessEqual(font[b"POINT_SIZE"], 24.0)
		self.failUnlessEqual(font[b"RESOLUTION_X"], 75)
		self.failUnlessEqual(font[b"RESOLUTION_Y"], 75)
		self.failUnlessEqual(font.get_comments(), [
				b"This is a sample font in 2.1 format."
			])
		self.failUnlessEqual(len(font.glyphs), 2)
		# Our code ignores PIXEL_SIZE but adds FACE_NAME, so the total is still
		# 19.
		self.failUnlessEqual(len(font.properties), 19)

	def test_basic_operation(self):
		testFontData = BytesIO(SAMPLE_FONT)
		testFont = reader.read_bdf(testFontData)

		self._check_font(testFont)

	def test_extra_blank_lines(self):
		"""
		We should ignore any extra lines in the input.
		"""
		testFontData = BytesIO(SAMPLE_FONT.replace(b"\n", b"\n\n"))
		testFont = reader.read_bdf(testFontData)

		self._check_font(testFont)

	def test_fractional_point_size(self):
		"""
		We should correctly interpret and store a fractional point size.
		"""
		bdf_data = (
				b"STARTFONT 2.1\n"
				b"FONT TestFont\n"
				b"SIZE 12.2 100 100\n"
				b"FONTBOUNDINGBOX 0 0 0 0\n"
				b"STARTPROPERTIES 5\n"
				b"FACE_NAME \"TestFont\"\n"
				b"FONT_ASCENT 0\n"
				b"FONT_DESCENT 0\n"
				b"RESOLUTION_X 100\n"
				b"RESOLUTION_Y 100\n"
				b"ENDPROPERTIES\n"
				b"CHARS 0\n"
				b"ENDFONT\n"
			)

		font = reader.read_bdf(BytesIO(bdf_data))

		self.failUnlessEqual("%0.1f" % font[b"POINT_SIZE"], "12.2")

	def test_ignored_properties(self):
		"""
		Certain properties can't be set.

		These properties include:
		 - FACE_NAME: comes from FONT header.
		 - POINT_SIZE: comes from SIZE header.
		 - RESOLUTION_X: comes from SIZE header.
		 - RESOLUTION_Y: comes from SIZE header.
		 - PIXEL_SIZE: calculated from 3 previous properties.
		"""

		bdf_data = (
				b"STARTFONT 2.1\n"
				b"FONT TestFont\n"
				b"SIZE 1 2 3\n"
				b"FONTBOUNDINGBOX 0 0 0 0\n"
				b"STARTPROPERTIES 5\n"
				b"FACE_NAME \"NotATestFont\"\n"
				b"POINT_SIZE 456\n"
				b"PIXEL_SIZE 789\n"
				b"RESOLUTION_X 012\n"
				b"RESOLUTION_Y 345\n"
				b"ENDPROPERTIES\n"
				b"CHARS 0\n"
				b"ENDFONT\n"
			)

		font = reader.read_bdf(BytesIO(bdf_data))

		self.failUnlessEqual(font[b"FACE_NAME"], b"TestFont")
		self.failUnlessEqual("%0.1f" % font[b"POINT_SIZE"], "1.0")
		self.failUnlessEqual(font[b"RESOLUTION_X"], 2)
		self.failUnlessEqual(font[b"RESOLUTION_Y"], 3)
		self.failIf(b"PIXEL_SIZE" in font)
