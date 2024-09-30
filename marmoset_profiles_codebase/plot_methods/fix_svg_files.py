import os


def fix_svg(dir_path):
    for fname in os.listdir(dir_path):
        if fname.endswith(".svg"):
            print("FIX", fname)
            path = os.path.join(dir_path, fname)
            f = open(path, "r")
            svg_text = f.read()
            f.close()

            svg_text = svg_text.replace(
                r"""style="font: 8px 'sans-serif'""", r"""style="font: 8px 'Arial'"""
            )

            svg_text = svg_text.replace(
                r"""style="font: 8px 'DejaVu Sans'""", r"""style="font: 8px 'Arial'"""
            )

            svg_text = svg_text.replace(
                r"""style="font: 4px 'sans-serif'""", r"""style="font: 4px 'Arial'"""
            )

            svg_text = svg_text.replace(
                r"""style="font: 4px 'DejaVu Sans'""", r"""style="font: 4px 'Arial'"""
            )

            new_f = open(path, "w")
            new_f.write(svg_text)
            new_f.close()
