# rust 2 python
'''
    use std::cmp::Ordering;
    let num1 = 10;
    let num2 = 10;
    match num1.cmp(&num2) {
        Ordering::Less => println!("Less"),
        Ordering::Equal => println!("Equal"),
        Ordering::Greater => println!("Greater"),
    };
'''
num1, num2 = 10, 10
if num1 > num2:
    print("great")
elif num1 == num2:
    print("equal")
else:
    print("less")

'''
    let num3 = "3";
    let guess : u32 = num3.trim().parse().expect("failed to parse guess to a value");
    println!("guess:{guess}");
'''
try:
    guess = int("3")
    print(f"guess:{guess}")
except:
    print("failed to parse guess to a value")

'''
let guess2 : u32 = match num3.trim().parse() {
        Ok(num) => num,
        Err(_) => 0,
    };
println!("guess2:{guess2}")
'''
try:
    guess = int("p3")
    print(f"guess:{guess}")
except Exception as e:
    print("failed to parse guess to a value, err:", e)

'''
let tuple1 : (i32, f64) = (1, 2.0);
'''
tuple1 = (1, 2.0)
print(f"tuple1:{tuple1}, type:{type(tuple1)}")

'''
    let tup = (500, 6.4, 1);
    let (x, y, z) = tup;
    println!("The value of y is: {y}");
'''

x, y, z = (500, 6.4, 1)
print(f"x:{x}, y:{y}, z:{z}")

'''
    let condition = true;
    let num4 = if condition {1} else {0};
    println!{"num4:{num4}"}
'''
num = 1 if True else 0
print(f"num:{num}")

'''
    let a = [10, 20, 30, 40, 50];
    for num in a {
        println!("num:{num}");
    }
'''
for num in [10, 20, 30, 40, 50]:
    print(f"num:{num}")

'''
    for num in (1..3).rev() {
        println!("num:{num}")
    }
'''
for num in reversed([10, 20, 30, 40, 50]):
    print(f"num:{num}")