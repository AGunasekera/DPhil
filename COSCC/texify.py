def texify(equations, filename):
    with open(filename + ".tex", "w") as f:
        f.write("\documentclass{minimal}")
        f.write("\n\n")
        f.write("\\begin{document}")
        for equation in equations:
            f.write("\n")
            texifySingleEquation(equation, f)
            f.write("\n")
        f.write("\n")
        f.write("\end{document}")

def texifySingleEquation(equation, file):
    file.write("$$")
    file.write(equation.__str__())
    file.write("$$")