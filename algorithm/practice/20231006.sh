#!/bin/bash
myArray=("item1" "item2" "item3")
echo "${myArray[@]}"
echo "${myArray[1]}"

if [ -f /dev/null ]; then
  echo "file"
elif [ -d /dev/null ]; then
  echo "dir"
elif [ -e /dev/null ]; then
  echo "exist"
else
  echo "else nothing"
fi

str1="Hello"
str2="world"
str3="Hello"
if [ $str1 = $str2 ]; then
  echo "str1 and str2 equal"
elif [ $str1 = $str3 ]; then
  echo "str1 is equal to str3"
else
  echo "not equal"
fi

# number compare
num1=1
num2=2
num3=1
if [ $num1 -eq $num3 ]; then
  echo "$num1 == $num3"
fi

if [ $num1 -lt $num2 ]; then
  echo "$num1 <= $num2"
fi

echo " last state: $?"

num=0
target=10
while [ $num -lt $target ]; do
    echo $num
    num=$(($num + 1))
done

num=`expr $num + 1`
echo "num:$num"

USAGE="usage"
if [ $# -lt 2 ]; then
  echo $USAGE
fi

case "$1" in
  -t) TARGS="-tvf $2" ;;
  -c) TARGS="-cvf $2.tar $2" ;;
  *) echo "$USAGE"
    exit 0
    ;;
esac
echo "TARGS $TARGS"