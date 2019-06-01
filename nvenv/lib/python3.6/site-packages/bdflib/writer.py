# bdflib, a library for working with BDF font files
# Copyright (C) 2009, Timothy Alle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math

def _quote_property_value(val):
	if isinstance(val, int):
		return b"%d" % val
	else:
		return b'"' + bytes(val).replace(b'"', b'""') + b'"'

def write_bdf(font, stream):
	"""
	Write a BDF-format font to the given stream.

	:param Font font: The font to write to the given stream.
	:param stream: The stream that will receive the font.

	``stream`` must be an object with at ``.write()`` method that takes a
	:class:`bytes`. If you want to write to an actual file, make sure you
	use the 'b' flag::

		bdflib.writer.write_bdf(font, open(path, 'wb'))
	"""
	# The font bounding box is the union of glyph bounding boxes.
	font_bbX = 0
	font_bbY = 0
	font_bbW = 0
	font_bbH = 0
	for g in font.glyphs:
		new_bbX = min(font_bbX, g.bbX)
		new_bbY = min(font_bbY, g.bbY)
		new_bbW = max(font_bbX + font_bbW, g.bbX + g.bbW) - new_bbX
		new_bbH = max(font_bbY + font_bbH, g.bbY + g.bbH) - new_bbY

		(font_bbX, font_bbY, font_bbW, font_bbH) = (
				new_bbX, new_bbY, new_bbW, new_bbH)

	# Calculated properties that aren't in the font model.
	properties = {
			b"PIXEL_SIZE": int(math.ceil(
				font[b"RESOLUTION_Y"] * font[b"POINT_SIZE"] / 72.0)),
			b"FONT_ASCENT": font_bbY + font_bbH,
			b"FONT_DESCENT": font_bbY * -1,
		}
	if len(font.glyphs_by_codepoint) > 0:
		properties[b"DEFAULT_CHAR"] = max(font.glyphs_by_codepoint.keys())
	properties.update(font.properties)

	# The POINT_SIZE property is actually in deci-points.
	properties[b"POINT_SIZE"] = int(properties[b"POINT_SIZE"] * 10)

	# Write the basic header.
	stream.write(b"STARTFONT 2.1\n")
	stream.write(b"FONT ")
	stream.write(font[b"FACE_NAME"])
	stream.write(b"\n")
	stream.write(b"SIZE %g %d %d\n" %
			(font[b"POINT_SIZE"], font[b"RESOLUTION_X"], font[b"RESOLUTION_Y"]))
	stream.write(b"FONTBOUNDINGBOX %d %d %d %d\n"
			% (font_bbW, font_bbH, font_bbX, font_bbY))

	# Write the properties
	stream.write(b"STARTPROPERTIES %d\n" % (len(properties),))
	keys = sorted(properties.keys())
	for key in keys:
		stream.write(key)
		stream.write(b" ")
		stream.write(_quote_property_value(properties[key]))
		stream.write(b"\n")
	stream.write(b"ENDPROPERTIES\n")

	# Write out the glyphs
	stream.write(b"CHARS %d\n" % (len(font.glyphs),))
	for glyph in font.glyphs:
		scalable_width = int(1000.0 * glyph.advance
				/ properties[b"PIXEL_SIZE"])
		stream.write(b"STARTCHAR ")
		stream.write(glyph.name)
		stream.write(b"\n")
		stream.write(b"ENCODING %d\n" % (glyph.codepoint,))
		stream.write(b"SWIDTH %d 0\n" % (scalable_width,))
		stream.write(b"DWIDTH %d 0\n" % (glyph.advance,))
		stream.write(b"BBX %d %d %d %d\n"
				% (glyph.bbW, glyph.bbH, glyph.bbX, glyph.bbY))
		stream.write(b"BITMAP\n")
		for row in glyph.get_data():
			stream.write(row)
			stream.write(b"\n")
		stream.write(b"ENDCHAR\n")

	stream.write(b"ENDFONT\n")
