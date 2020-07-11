f = open("mid1.txt", "r", encoding="utf-8")
for each in f:
    each = each.rstrip()
    parts = each.split(" ")
    if len(parts) != 3:
        print(parts)
