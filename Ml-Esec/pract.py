li = [1,4,7,10]
t = 2
n = 4
res = 0
counter = 0
while counter < max(li) + t:
    if counter in li :
        print(counter in li)
        time = counter + t
        while counter < time:
            res += 1
            print(res,counter)
            if counter in li:
                time = time + t
            counter += 1
        time = 0
    else:
        counter += 1

print(res)
