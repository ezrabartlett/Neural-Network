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
Automatically generate visual variants of a font.
"""

def embolden(font, maintain_spacing=True):
	"""
	Create a bold version of a font by widening each glyph.

	:param Font font: The font to embolden.
	:param bool maintain_spacing: If true, each glyph's advance width
		will be incremented, because a wider glyph needs more room.
	:returns: A copy of ``font``, with each glyph emboldened.
	:rtype: :class:`.Font`

	To embolden a glyph, it is drawn over itself one pixel to the right,
	making each vertical stroke one pixel wider.
	"""
	res = font.copy()

	for cp in res.codepoints():
		g = res[cp]
		g.merge_glyph(g, 1,0)
		if maintain_spacing:
			g.advance += 1

	return res


def merge(base, custom):
	"""
	Create a new font by choosing glyphs from two other fonts.

	:param Font base: The lower-priority font.
	:param Font custom: The higher-priority font.
	:returns: A font where each glyph comes from ``base`` or ``custom``.

	For any given codepoint, the resulting font will use the corresponding
	glyph from ``custom`` if there is one, falling back to the glyph from
	``base``. The new font's properties and other metadata are all copied
	from ``custom``.
	"""
	res = custom.copy()

	for cp in base.codepoints():
		if cp not in res:
			old_glyph = base[cp]
			res.new_glyph_from_data(old_glyph.name, old_glyph.get_data(),
					old_glyph.bbX, old_glyph.bbY, old_glyph.bbW, old_glyph.bbH,
					old_glyph.advance, old_glyph.codepoint)

	return res
