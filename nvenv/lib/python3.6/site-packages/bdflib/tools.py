"""
Command-line tools for common bdflib operations.
"""
import argparse
import bdflib
import unicodedata
from bdflib import reader, writer, effects, glyph_combining


def add_standard_arguments(parser):
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=bdflib.__version__,
    )

    # I'd love to use argparse.FileType, but https://bugs.python.org/issue14156
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Read a BDF font from file INPUT.",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT",
        help="Write the resulting BDF font to file OUTPUT.",
    )


def embolden():
    parser = argparse.ArgumentParser(
        description="Add a faux-bold effect to a font.",
    )
    add_standard_arguments(parser)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--maintain-spacing",
            dest="maintain_spacing",
            action="store_true",
            default="True",
            help="Expand each character's spacing to account for emboldening "
                "(default)",
                       )
    group.add_argument("--ignore-spacing",
            dest="maintain_spacing",
            action="store_false",
            help="Let bold characters use their original spacing",
                       )

    args = parser.parse_args()

    with open(args.input, "rb") as input:
        with open(args.output, "wb") as output:
            font = reader.read_bdf(input)
            bold = effects.embolden(font, args.maintain_spacing)
            writer.write_bdf(bold, output)


def fill():
    parser = argparse.ArgumentParser(
        description="Generate pre-composed glyphs from available components",
    )
    add_standard_arguments(parser)
    args = parser.parse_args()

    with open(args.input, "rb") as input:
        with open(args.output, "wb") as output:
            print("Reading font...")
            font = reader.read_bdf(input)
            print("Building list of decompositions...")
            decompositions = glyph_combining.build_unicode_decompositions()
            print("Generating combined characters...")
            filler = glyph_combining.FontFiller(font, decompositions)
            filler.add_decomposable_glyphs_to_font()
            print("Writing out result...")
            writer.write_bdf(font, output)

            # Show the inventory of things this font is missing.
            print()
            filler.unknown_classes.show()
            print()
            filler.missing_chars.show(
                lambda char: "%r (%s)" % (char, unicodedata.name(char))
            )


def merge():
    parser = argparse.ArgumentParser(
        description="""
            For each code-point that has a glyph in BASE or CUSTOM,
            the resulting font will use the glyph from CUSTOM if present,
            otherwise falling back to the glyph in BASE.
        """,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=bdflib.__version__,
    )

    # I'd love to use argparse.FileType, but https://bugs.python.org/issue14156
    parser.add_argument(
        "base",
        metavar="BASE",
        help="Path to a BDF font file, used for fallback glyphs.",
    )
    parser.add_argument(
        "custom",
        metavar="CUSTOM",
        help="Path to a BDF font file, whose glyphs override those in BASE.",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT",
        help="The resulting font will be written to the file OUTPUT.",
    )
    args = parser.parse_args()

    with open(args.base, "rb") as base:
        with open(args.custom, "rb") as custom:
            with open(args.output, "wb") as output:
                base_font = reader.read_bdf(base)
                custom_font = reader.read_bdf(custom)
                merged = effects.merge(base_font, custom_font)
                writer.write_bdf(merged, output)


def passthrough():
    parser = argparse.ArgumentParser(
        description="Parse and re-serialise a BDF font.",
    )
    add_standard_arguments(parser)
    args = parser.parse_args()

    with open(args.input, "rb") as input:
        with open(args.output, "wb") as output:
            font = reader.read_bdf(input)
            writer.write_bdf(font, output)
