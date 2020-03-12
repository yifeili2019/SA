create_file = open("/home/dell/test1/src/gdp/src/bbox.txt",'w')
for i in range(10):
    temp = str(i)
    create_file.write(temp)
    create_file.write("\n")

create_file.close()