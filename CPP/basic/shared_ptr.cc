
class MyCar:public std::enable_shared_from_this<MyCar>
{
public:
  shared_ptr<MyCar> get_ptr() {
    return shared_from_this();
  }
  ~MyCar() {
    std::cout << "free ~Mycar()" << std::endl;
  }
private:

};

int main()
{
  MyCar* _myCar = new MyCar();
  shared_ptr<MyCar> _myCar1(_myCar);
  shared_ptr<MyCar> _myCar2 = _myCar->get_ptr();
  std::cout << _myCar1.use_count() << std::endl;
  std::cout << _myCar2.use_count() << std::endl;
  return 0;
}