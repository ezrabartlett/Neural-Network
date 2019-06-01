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

"""
Classes that represent a bitmap font, with its glyphs and metadata.

.. testsetup::

	from bdflib.model import *
"""
# There are more reliable sources than BDF properties for these settings, so
# we'll ignore attempts to set them.
IGNORABLE_PROPERTIES = [
		b"FACE_NAME",
		b"POINT_SIZE",
		b"PIXEL_SIZE",
		b"RESOLUTION_X",
		b"RESOLUTION_Y",
	]


class GlyphExists(Exception):
	"""
	Raised when	creating a new glyph for a codepoint that already has one.
	"""


class Glyph(object):
	"""
	Represents a font glyph and associated properties.

	:param bytes name: The name of this glyph, ASCII encoded.
	:param data: the bitmap data for the glyph, as ASCII encoded hexadecimal
		digits.
	:type data: iterable of :class:`bytes`
	:param int bbX: The left-most edge of the glyph's bounding box, in
		pixels.
	:param int bbY: The bottom-most edge of the glyph's bounding box, in pixels.
	:param int bbW: The glyph's bounding-box extends this many pixels right of
		``bbX`` (must be >= 0).
	:param int bbH: The glyph's bounding-box extends this many pixels upward
		from ``bbY`` (must be >= 0).
	:param int advance: After drawing this glyph, the next glyph will be
		drawn this many pixels to the right.
	:param int codepoint: The Unicode codepoint that this glyph represents.

	The glyph bitmap data is an iterable of ``bytes``, where the first item
	represents the top row of pixels, the next item represents the next row
	of pixels, and so forth. There should be ``bbH`` items in total. Each
	item is one or more hexadecimal digits representing the pixels of that
	row: the first digit represents the leftmost four pixels (most
	significant bit to the left), the next digit represents the next four
	pixels, and so forth. If ``bbW`` is not an integer multiple of 4, the
	row should be padded with 0 bits.

	For example, let's say we want to render a capital A, like so:

	.. code-block:: text

		.###.
		#...#
		#####
		#...#

	We encode that pattern in binary, pad to a multiple of four bits, and
	convert to hexadecimal::

		>>> raw_data = [
		...     0b01110000,
		...     0b10001000,
		...     0b11111000,
		...     0b10001000,
		... ]
		>>> for each in raw_data:
		...     print("{:X}".format(each))
		70
		88
		F8
		88

	Therefore, to create a glyph representing that image::

		>>> my_glyph = Glyph(
		...     name=b"capital A",
		...     data=[b'70', b'88', b'F8', b'88'],
		...     bbW=5,
		...     bbH=4,
		...     advance=6,
		...     codepoint=65,
		... )

	To aid debugging, the ``Glyph`` class has a ``__str__`` method that
	renders the glyph on a grid, with the glyph origin marked::

		>>> print(my_glyph)
		|###.
		#...#
		#####
		#---#

	The glyph origin isn't actually visible because it's underneath the lower
	left corner of the glyph, but you can see the X axis across the bottom
	and the Y axis up the left.

	.. py:attribute:: advance

		(:class:`int`)
		How far to the right the next glyph should be drawn, in pixels.

	.. py:attribute:: data

		(:class:`list` of :class:`int`) Glyph bitmap data.

		Unlike the format passed to the constructor and returned from
		:meth:`get_data`, this uses a more Python-friendly format::

			>>> my_glyph = Glyph(
			...     name=b"capital A",
			...     data=[b'70', b'88', b'F8', b'88'],
			...     bbW=5,
			...     bbH=4,
			... )
			>>> for row in my_glyph.data:
			...     print("{:05b}".format(row))
			10001
			11111
			10001
			01110

		Each item of the ``.data`` property is a :class:`int` `bbW` bits wide,
		representing the pixels of a single row. The first item in ``.data`` is
		the lowest row in the glyph, so that list indices increase in the same
		direction as pixel coordinates.

		If you want to get the actual coordinates of the glyph's drawn pixels
		(for example, to draw the glyph onto an image), the easiest way is to
		use a loop like this::

			ink_coords = [
				(x, y)
				for x in range(my_glyph.bbW)
				for y in range(my_glyph.bbH)
				if my_glyph.data[y] & (1 << (my_glyph.bbW - x - 1)
			]

		You might also want to look at :meth:`iter_pixels`.
	"""

	def __init__(self, name, data=None, bbX=0, bbY=0, bbW=0, bbH=0,
			advance=0, codepoint=None):
		self.name = name
		self.bbX = bbX
		self.bbY = bbY
		self.bbW = bbW
		self.bbH = bbH
		if data is None:
			self.data = []
		else:
			self._set_data(data)
		self.advance = advance
		if codepoint is None:
			self.codepoint = -1
		else:
			self.codepoint = codepoint

	def __str__(self):
		def padding_char(x,y):
			if x == 0 and y == 0:
				return '+'
			elif x == 0:
				return '|'
			elif y == 0:
				return '-'
			else:
				return '.'

		# What are the extents of this bitmap, given that we always want to
		# include the origin?
		bitmap_min_X = min(0, self.bbX)
		bitmap_max_X = max(0, self.bbX + self.bbW-1)
		bitmap_min_Y = min(0, self.bbY)
		bitmap_max_Y = max(0, self.bbY + self.bbH-1)

		res = []
		for y in range(bitmap_max_Y, bitmap_min_Y - 1, -1):
			res_row = []
			# Find the data row associated with this output row.
			if self.bbY <= y < self.bbY + self.bbH:
				data_row = self.data[y - self.bbY]
			else:
				data_row = 0
			for x in range(bitmap_min_X, bitmap_max_X + 1):
				# Figure out which bit controls (x,y)
				bit_number = self.bbW - (x - self.bbX) - 1
				# If we're in a cell covered by the bitmap and this particular
				# bit is set...
				if self.bbX <= x < self.bbX + self.bbW and (
						data_row >> bit_number & 1):
					res_row.append('#')
				else:
					res_row.append(padding_char(x,y))
			res.append("".join(res_row))

		return "\n".join(res)

	def _set_data(self, data):
		self.data = []
		for row in data:
			paddingbits = len(row) * 4 - self.bbW
			self.data.append(int(row, 16) >> paddingbits)

		# Make the list indices match the coordinate system
		self.data.reverse()

	def get_data(self):
		"""
		Returns the glyph's bitmap data.

		This may use a different amount of padding than the value originally
		passed to the constructor, and hexadecimal digits may be upper-case
		instead of lower-case, but the meaning should be the same.

		See also the :meth:`iter_pixels` method,
		which gives you a ``bool`` for each pixel,
		and the underlying storage in the :attr:`data` property.

		:returns: Hex-encoded bitmap data
		:rtype: :class:`list` of :class:`bytes`
		"""
		res = []

		# How many bytes do we need to represent the bits in each row?
		rowWidth, extraBits = divmod(self.bbW, 8)

		# How many bits of padding do we need to round up to a full byte?
		if extraBits > 0:
			rowWidth += 1
			paddingBits = 8 - extraBits
		else:
			paddingBits = 0

		for row in self.data:
			# rowWidth is the number of bytes, but Python wants the number of
			# nybbles, so multiply by 2.
			res.append(b"%0*X" % (rowWidth*2, row << paddingBits))

		# self.data goes bottom-to-top like any proper coordinate system does,
		# but res wants to be top-to-bottom like any proper stream-output.
		res.reverse()

		return res

	def get_bounding_box(self):
		"""
		Returns the position and dimensions of the glyph's bounding box.

		:returns: The left, bottom, width and height of the bounding box, as
			passed to the constructor.
		:rtype: :class:`tuple` of :class:`int`, :class:`int`, :class:`int`,
			:class:`int`
		"""
		return (self.bbX, self.bbY, self.bbW, self.bbH)

	def merge_glyph(self, other, atX, atY):
		"""
		Draw another glyph onto this one at the given coordinates.

		:param Glyph other: The other glyph to draw onto this one.
		:param int atX: The other glyph's origin will be placed at this X
			offset in this glyph.
		:param int atY: The other glyph's origin will be placed at this Y
			offset in this glyph.

		This glyph's bounding box will be stretch to include the area of
		the added glyph, but the :attr:`advance` will not be modified.
		"""
		# Calculate the new metrics
		new_bbX = min(self.bbX, atX + other.bbX)
		new_bbY = min(self.bbY, atY + other.bbY)
		new_bbW = max(self.bbX + self.bbW,
				atX + other.bbX + other.bbW) - new_bbX
		new_bbH = max(self.bbY + self.bbH,
				atY + other.bbY + other.bbH) - new_bbY

		# Calculate the new data
		new_data = []
		for y in range(new_bbY, new_bbY + new_bbH):
			# If the old glyph has a row here...
			if self.bbY <= y < self.bbY + self.bbH:
				old_row = self.data[y-self.bbY]

				# If the right-hand edge of the bounding box has moved right,
				# we'll need to left shift the old-data to get more empty space
				# to draw the new glyph into.
				right_edge_delta = (new_bbX + new_bbW) - (self.bbX + self.bbW)
				if right_edge_delta > 0:
					old_row <<= right_edge_delta
			else:
				old_row = 0
			# If the new glyph has a row here...
			if atY + other.bbY <= y < atY + other.bbY + other.bbH:
				new_row = other.data[y - other.bbY - atY]

				# If the new right-hand-edge ofthe bounding box
				if atX + other.bbX + other.bbW < new_bbX + new_bbW:
					new_row <<= ((new_bbX + new_bbW)
							- (atX + other.bbX + other.bbW))
			else:
				new_row = 0
			new_data.append(old_row | new_row)

		# Update our properties with calculated values
		self.bbX = new_bbX
		self.bbY = new_bbY
		self.bbW = new_bbW
		self.bbH = new_bbH
		self.data = new_data

	def get_ascent(self):
		"""
		Returns the distance from the Y axis to the highest point of the glyph.

		This is zero if no part of the glyph is above the Y axis.

		:returns: The ascent of this glyph.
		:rtype: :class:`int`
		"""
		res = self.bbY + self.bbH

		# Each empty row at the top of the bitmap should not be counted as part
		# of the ascent.
		for row in self.data[::-1]:
			if row != 0:
				break
			else:
				res -= 1

		return res

	def get_descent(self):
		"""
		Returns the distance from the Y axis to the lowest point of the glyph.

		This is zero if no part of the glyph is below the Y axis.

		:returns: The descent of this glyph.
		:rtype: :class:`int`
		"""
		res =  -1 * self.bbY

		# Each empty row at the bottom of the bitmap should not be counted as
		# part of the descent.
		for row in self.data:
			if row != 0:
				break
			else:
				res -= 1

		return res

	def iter_pixels(self):
		"""
		Yields the state of pixels within the bounding box.

		This method returns an iterable of ``bbH`` rows,
		from the top of the glyph (large X values)
		to the bottom (small X values).
		Each row is an iterable of ``bbW`` booleans,
		from left to right.
		Each boolean is ``True`` if that pixel should be drawn,
		and otherwise ``False``.

		Alternatively,
		you can obtain the glyph data in BDF format with :meth:`get_data()`,
		or access the raw bitmap via the :attr:`data` property.

		:returns: the state of each pixel
		:rtype: iterable of iterable of :class:`bool`
		"""
		return (
			(bool(row & (1 << self.bbW - x - 1)) for x in range(self.bbW))
			for row in (self.data[self.bbH - y - 1] for y in range(self.bbH))
		)


class Font(object):
	"""
	Represents the entire font and font-global properties.

	:param bytes name: The human-readable name of this font, ASCII encoded.
	:param int ptSize: The nominal size of this font in PostScript points (1/72
		of an inch).
	:param int xdpi: The horizontal resolution of this font in dots-per-inch.
	:param int ydpi: The vertical resolution of this font in dots-per-inch.

	Instances of this class can be used like :class:`dict` instances.
	:class:`bytes` keys refer to the font's properties and are associated with
	:class:`bytes` instances, while :class:`int` keys refer to the code-points
	the font supports, and are associated with :class:`Glyph` instances.

		>>> myfont = Font(
		...     b"My Font",
		...     ptSize=12,
		...     xdpi=96,
		...     ydpi=96,
		... )
		>>> myfont[b"POINT_SIZE"]
		12
		>>> a_glyph = myfont.new_glyph_from_data(
		...     "capital A",
		...     codepoint=65,
		... )
		>>> a_glyph == myfont[65]
		True

	.. note::

		Some properties (the name, point-size and resolutions) are
		required, and although they can be examined via the ``dict`` interface,
		they cannot be removed with the ``del`` statement.
	"""

	#: (iterable of :class:`Glyph`) All the glyphs in this font, even the ones
	#: with no associated codepoint.
	glyphs = []

	def __init__(self, name, ptSize, xdpi, ydpi):
		"""
		Initialise this font object.
		"""
		self.properties = {
				b"FACE_NAME": bytes(name),
				b"POINT_SIZE": ptSize,
				b"RESOLUTION_X": xdpi,
				b"RESOLUTION_Y": ydpi,
			}
		self.glyphs = []
		self.glyphs_by_codepoint = {}
		self.comments = []

	def add_comment(self, comment):
		"""
		Add one or more lines of text to the font's comment field.

		:param bytes comment: Human-readable text to add to the
			comment, ASCII encoded. It may include newline characters.

		The added text will begin on a new line.
		"""
		lines = bytes(comment).split(b"\n")
		self.comments.extend(lines)

	def get_comments(self):
		"""
		Retrieve the lines of the font's comment field.

		:returns: The comment text, ASCII encoded.
		:rtype: :class:`list` of :class:`bytes`
		"""
		return self.comments

	def __setitem__(self, name, value):
		assert isinstance(name, bytes)
		if name not in IGNORABLE_PROPERTIES:
			self.properties[name] = value

	def __getitem__(self, key):
		if isinstance(key, bytes):
			return self.properties[key]
		elif isinstance(key, int):
			return self.glyphs_by_codepoint[key]

	def __delitem__(self, key):
		if key in IGNORABLE_PROPERTIES: return
		elif isinstance(key, bytes):
			del self.properties[key]
		elif isinstance(key, int):
			g = self.glyphs_by_codepoint[key]
			self.glyphs.remove(g)
			del self.glyphs_by_codepoint[key]

	def __contains__(self, key):
		if isinstance(key, bytes):
			return key in self.properties
		elif isinstance(key, int):
			return key in self.glyphs_by_codepoint

	def new_glyph_from_data(self, name, data=None, bbX=0, bbY=0, bbW=0, bbH=0,
			advance=0, codepoint=None):
		"""
		Add a new :class:`Glyph` to this font.

		This method's arguments are passed to the :class:`Glyph` constructor.

		If you include the ``codepoint`` parameter, the codepoint will be
		included in the result of :meth:`codepoints` and you will be able
		to look up the glyph by codepoint later. Otherwise, it will only be
		available via the :attr:`glyphs` property.

		:returns: the newly-created Glyph
		:rtype: :class:`Glyph`
		:raises GlyphExists: if an existing glyph is already associated with
			the requested codepoint.
		"""
		g = Glyph(name, data, bbX, bbY, bbW, bbH, advance, codepoint)
		self.glyphs.append(g)
		if codepoint is not None and codepoint >= 0:
			if codepoint in self.glyphs_by_codepoint:
				raise GlyphExists("A glyph already exists for codepoint %r"
						% codepoint)
			else:
				self.glyphs_by_codepoint[codepoint] = g
		return g

	def copy(self):
		"""
		Returns a deep copy of this font.

		The new font, along with all of its properties and glyphs, may be
		modified without affecting this font.

		:returns: A new, independent copy of this Font
		:rtype: :class:`Font`
		"""

		# Create a new font object.
		res = Font(
				self[b"FACE_NAME"],
				self[b"POINT_SIZE"],
				self[b"RESOLUTION_X"],
				self[b"RESOLUTION_Y"],
			)

		# Copy the comments across.
		for c in self.comments:
			res.add_comment(c)

		# Copy the properties across.
		for p in self.properties:
			res[p] = self[p]

		# Copy the glyphs across.
		for g in self.glyphs:
			res.new_glyph_from_data(g.name, g.get_data(), g.bbX, g.bbY, g.bbW,
					g.bbH, g.advance, g.codepoint)

		return res

	def property_names(self):
		"""
		Returns the names of this font's properties.

		These names can be used with the regular dict syntax to retrieve the
		associated value.

		:returns: Property names
		:rtype: iterable of :class:`bytes`
		"""
		return self.properties.keys()

	def codepoints(self):
		"""
		Returns the codepoints that this font has glyphs for.

		These codepoints can be used with the regular dict syntax to retrieve
		the associated glyphs

		:returns: Supported codepoints
		:rtype: iterable of :class:`Glyph`
		"""
		return self.glyphs_by_codepoint.keys()
