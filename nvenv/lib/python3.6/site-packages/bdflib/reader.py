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

from bdflib import model

def _read_glyph(iterable, font):
	glyphName = ""
	codepoint = -1
	bbX = 0
	bbY = 0
	bbW = 0
	bbH = 0
	advance = 0
	data = []

	for key, values in iterable:
		if key == b"STARTCHAR":
			glyphName = values
		elif key == b"ENCODING":
			codepoint = int(values.split()[0])
		elif key == b"DWIDTH":
			advance = int(values.split()[0])
		elif key == b"BBX":
			bbW, bbH, bbX, bbY = map(int, values.split())
		elif key == b"BITMAP":
			# The next bbH lines describe the font bitmap.
			data = [next(iterable)[0] for _ in range(bbH)]
			assert next(iterable)[0] == b"ENDCHAR"
			break

	font.new_glyph_from_data(glyphName, data, bbX, bbY, bbW, bbH, advance,
			codepoint)


def _unquote_property_value(value):
	# Python 2.x compat: we can't predict whether indexing into a bytestring
	# will give a one-char string or an integer.
	if value[0] == b'"'[0]:
		# Must be a string. Remove the outer quotes and un-escape embedded
		# quotes.
		return value[1:-1].replace(b'""', b'"')
	else:
		# No quotes, must be an integer.
		return int(value)

def _next_token(iterable, comments):
	"""
	Yields only non-whitespace, non-comment lines.
	"""
	for line in iterable:
		parts = iter(line.strip().split(None, 1))
		key = next(parts, None)
		if not key:
			continue
		elif key == b"COMMENT":
			comments.append(next(parts, b""))
		else:
			yield key, next(parts, None)

def read_bdf(raw_iterable):
	"""
	Read a BDF-format font from the given source.

	:param raw_iterable: Each item should be a single line of the BDF file,
		ASCII encoded.
	:type raw_iterable: iterable of :class:`bytes`
	:returns: the resulting font object
	:rtype: :class:`.Font`

	If you want to read an actual file, make sure you use the 'b' flag so
	you get bytes instead of text::

		font = bdflib.reader.read_bdf(open(path, 'rb'))
	"""
	name = b""
	pointSize = 0.0
	resX = 0
	resY = 0
	comments = []
	font = None

	iterable = _next_token(raw_iterable, comments)
	for key, values in iterable:
		if key == b"FONT":
			name = values
		elif key == b"SIZE":
			pointSize, resX, resY = values.split()
			pointSize = float(pointSize)
			resX = int(resX)
			resY = int(resY)
		elif key == b"FONTBOUNDINGBOX":
			# We don't care about the font bounding box, but it's the last
			# header to come before the variable-length fields for which we
			# need a font object around.
			font = model.Font(name, pointSize, resX, resY)
		elif key == b"STARTPROPERTIES":
			assert font is not None

			propertyCount = int(values)
			for _ in range(propertyCount):
				key, values = next(iterable)
				font[key] = _unquote_property_value(values)

			key, values = next(iterable)
			assert key == b"ENDPROPERTIES"
		elif key == b"CHARS":
			for _ in range(int(values)):
				_read_glyph(iterable, font)
			break

	assert next(iterable)[0] == b"ENDFONT"

	# Set font comments
	for c in comments:
		font.add_comment(c)

	return font
