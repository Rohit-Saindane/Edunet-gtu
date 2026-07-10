my_list = [1, 2, 3]
my_list.append(4)
my_list.extend([5, 6])
my_list.insert(1, 10)
my_list.remove(3)
popped_val = my_list.pop()

my_tuple = (1, 2, 2, 3, 4)
count_two = my_tuple.count(2)
idx_three = my_tuple.index(3)
t_len = len(my_tuple)
t_min = min(my_tuple)
t_max = max(my_tuple)

my_set = {1, 2, 3}
my_set.add(4)
my_set.remove(2)
my_set.discard(5)
other_set = {3, 4, 5}
union_set = my_set.union(other_set)
inter_set = my_set.intersection(other_set)

my_dict = {"a": 1, "b": 2}
my_dict.update({"c": 3})
d_keys = list(my_dict.keys())
d_values = list(my_dict.values())
d_items = list(my_dict.items())
got_val = my_dict.get("a")

names = ["Alice", "Bob", "Charlie"]
for i in range(len(names)):
    print(names[i])
    print(type(names[i]))
    if names[i] == "Bob":
        break

for num in range(5):
    if num % 2 == 0:
        print(num)
    else:
        continue

for val in range(3):
    if val == 0:
        print("Zero")
    elif val == 1:
        print("One")
    else:
        print("Other")

user_input = input("Enter a number (or 'exit' to quit): ")
while user_input != "exit":
    if user_input.isdigit():
        num = int(user_input)
        if num > 0:
            if num % 2 == 0:
                print("Positive Even")
            else:
                print("Positive Odd")
        else:
            print("Non-positive")
    else:
        pass
    user_input = input("Enter a number (or 'exit' to quit): ")
