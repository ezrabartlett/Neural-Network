import unittest
from io import BytesIO

from bdflib import model, writer

class TestBDFWriter(unittest.TestCase):

	def setUp(self):
		self.font = model.Font(b"TestFont", 12, 100,100)

	def test_basic_writing(self):
		"""
		Writing out a simple font should work.
		"""
		self.font.new_glyph_from_data(b"TestGlyph", [b"4", b"8"], 0,0, 2,2, 3, 1)

		stream = BytesIO()
		writer.write_bdf(self.font, stream)

		self.failUnlessEqual(stream.getvalue(),
				b"STARTFONT 2.1\n"
				b"FONT TestFont\n"
				b"SIZE 12 100 100\n"
				b"FONTBOUNDINGBOX 2 2 0 0\n"
				b"STARTPROPERTIES 8\n"
				b"DEFAULT_CHAR 1\n"
				b"FACE_NAME \"TestFont\"\n"
				b"FONT_ASCENT 2\n"
				b"FONT_DESCENT 0\n"
				b"PIXEL_SIZE 17\n"
				b"POINT_SIZE 120\n"
				b"RESOLUTION_X 100\n"
				b"RESOLUTION_Y 100\n"
				b"ENDPROPERTIES\n"
				b"CHARS 1\n"
				b"STARTCHAR TestGlyph\n"
				b"ENCODING 1\n"
				b"SWIDTH 176 0\n"
				b"DWIDTH 3 0\n"
				b"BBX 2 2 0 0\n"
				b"BITMAP\n"
				b"40\n"
				b"80\n"
				b"ENDCHAR\n"
				b"ENDFONT\n"
			)

	def test_empty_font(self):
		"""
		We should be able to write an empty font.
		"""
		stream = BytesIO()
		writer.write_bdf(self.font, stream)

		self.failUnlessEqual(stream.getvalue(),
				b"STARTFONT 2.1\n"
				b"FONT TestFont\n"
				b"SIZE 12 100 100\n"
				b"FONTBOUNDINGBOX 0 0 0 0\n"
				b"STARTPROPERTIES 7\n"
				b"FACE_NAME \"TestFont\"\n"
				b"FONT_ASCENT 0\n"
				b"FONT_DESCENT 0\n"
				b"PIXEL_SIZE 17\n"
				b"POINT_SIZE 120\n"
				b"RESOLUTION_X 100\n"
				b"RESOLUTION_Y 100\n"
				b"ENDPROPERTIES\n"
				b"CHARS 0\n"
				b"ENDFONT\n"
			)

	def test_bounding_box_calculations(self):
		"""
		FONTBOUNDINGBOX should be calculated from individual glyphs.
		"""
		self.font.new_glyph_from_data(b"TestGlyph1", [b"4", b"8"],
				1,3, 2,2, 3, 1)
		self.font.new_glyph_from_data(b"TestGlyph2", [b"4", b"8"],
				-5,-7, 2,2, 1, 2)

		stream = BytesIO()
		writer.write_bdf(self.font, stream)

		self.failUnlessEqual(stream.getvalue(),
				b"STARTFONT 2.1\n"
				b"FONT TestFont\n"
				b"SIZE 12 100 100\n"
				b"FONTBOUNDINGBOX 8 12 -5 -7\n"
				b"STARTPROPERTIES 8\n"
				b"DEFAULT_CHAR 2\n"
				b"FACE_NAME \"TestFont\"\n"
				b"FONT_ASCENT 5\n"
				b"FONT_DESCENT 7\n"
				b"PIXEL_SIZE 17\n"
				b"POINT_SIZE 120\n"
				b"RESOLUTION_X 100\n"
				b"RESOLUTION_Y 100\n"
				b"ENDPROPERTIES\n"
				b"CHARS 2\n"
				b"STARTCHAR TestGlyph1\n"
				b"ENCODING 1\n"
				b"SWIDTH 176 0\n"
				b"DWIDTH 3 0\n"
				b"BBX 2 2 1 3\n"
				b"BITMAP\n"
				b"40\n"
				b"80\n"
				b"ENDCHAR\n"
				b"STARTCHAR TestGlyph2\n"
				b"ENCODING 2\n"
				b"SWIDTH 58 0\n"
				b"DWIDTH 1 0\n"
				b"BBX 2 2 -5 -7\n"
				b"BITMAP\n"
				b"40\n"
				b"80\n"
				b"ENDCHAR\n"
				b"ENDFONT\n"
			)

	def test_property_quoting(self):
		"""
		Test that property values are quoted properly.
		"""

		self.font[b"AN_INTEGER"] = 42
		self.font[b"A_STRING"] = b"42"
		self.font[b"STRING_WITH_QUOTES"] = b'Neville "The Banker" Robinson'

		stream = BytesIO()
		writer.write_bdf(self.font, stream)

		self.failUnlessEqual(stream.getvalue(),
				b"STARTFONT 2.1\n"
				b"FONT TestFont\n"
				b"SIZE 12 100 100\n"
				b"FONTBOUNDINGBOX 0 0 0 0\n"
				b"STARTPROPERTIES 10\n"
				b"AN_INTEGER 42\n"
				b"A_STRING \"42\"\n"
				b"FACE_NAME \"TestFont\"\n"
				b"FONT_ASCENT 0\n"
				b"FONT_DESCENT 0\n"
				b"PIXEL_SIZE 17\n"
				b"POINT_SIZE 120\n"
				b"RESOLUTION_X 100\n"
				b"RESOLUTION_Y 100\n"
				b"STRING_WITH_QUOTES \"Neville \"\"The Banker\"\" Robinson\"\n"
				b"ENDPROPERTIES\n"
				b"CHARS 0\n"
				b"ENDFONT\n"
			)

	def test_default_char_setting(self):
		"""
		If a default char is explicitly set, it should be used.
		"""
		self.font.new_glyph_from_data(b"TestGlyph1", [b"4", b"8"],
				0,0, 2,2, 3, 1)
		self.font.new_glyph_from_data(b"TestGlyph2", [b"8", b"4"],
				0,0, 2,2, 3, 0xFFFD)
		self.font[b"DEFAULT_CHAR"] = 0xFFFD

		stream = BytesIO()
		writer.write_bdf(self.font, stream)

		self.failUnlessEqual(stream.getvalue(),
				b"STARTFONT 2.1\n"
				b"FONT TestFont\n"
				b"SIZE 12 100 100\n"
				b"FONTBOUNDINGBOX 2 2 0 0\n"
				b"STARTPROPERTIES 8\n"
				b"DEFAULT_CHAR 65533\n"
				b"FACE_NAME \"TestFont\"\n"
				b"FONT_ASCENT 2\n"
				b"FONT_DESCENT 0\n"
				b"PIXEL_SIZE 17\n"
				b"POINT_SIZE 120\n"
				b"RESOLUTION_X 100\n"
				b"RESOLUTION_Y 100\n"
				b"ENDPROPERTIES\n"
				b"CHARS 2\n"
				b"STARTCHAR TestGlyph1\n"
				b"ENCODING 1\n"
				b"SWIDTH 176 0\n"
				b"DWIDTH 3 0\n"
				b"BBX 2 2 0 0\n"
				b"BITMAP\n"
				b"40\n"
				b"80\n"
				b"ENDCHAR\n"
				b"STARTCHAR TestGlyph2\n"
				b"ENCODING 65533\n"
				b"SWIDTH 176 0\n"
				b"DWIDTH 3 0\n"
				b"BBX 2 2 0 0\n"
				b"BITMAP\n"
				b"80\n"
				b"40\n"
				b"ENDCHAR\n"
				b"ENDFONT\n"
			)

	def test_resolution_calculations(self):
		"""
		The pixel size should be correctly calculated from the point size.
		"""
		tests = [
				(12, 72, 12),
				(12, 100, 17),
				(12.2, 100, 17),
				(12, 144, 24),
			]

		for pointSz, res, pixelSz in tests:
			deci_pointSz = int(pointSz * 10)

			font = model.Font(b"TestFont", pointSz, res, res)

			stream = BytesIO()
			writer.write_bdf(font, stream)

			self.failUnlessEqual(stream.getvalue(),
					b"STARTFONT 2.1\n"
					b"FONT TestFont\n"
					b"SIZE %(pointSz)g %(res)d %(res)d\n"
					b"FONTBOUNDINGBOX 0 0 0 0\n"
					b"STARTPROPERTIES 7\n"
					b"FACE_NAME \"TestFont\"\n"
					b"FONT_ASCENT 0\n"
					b"FONT_DESCENT 0\n"
					b"PIXEL_SIZE %(pixelSz)d\n"
					b"POINT_SIZE %(deci_pointSz)d\n"
					b"RESOLUTION_X %(res)d\n"
					b"RESOLUTION_Y %(res)d\n"
					b"ENDPROPERTIES\n"
					b"CHARS 0\n"
					b"ENDFONT\n"
					% {
						b"pointSz": pointSz,
						b"res": res,
						b"pixelSz": pixelSz,
						b"deci_pointSz": deci_pointSz,
					},
				)

