subroutine gs(x, s2, r, n)

    integer*4, intent(in) :: n
    real*8, intent(in), dimension(n) :: x
    real*8, intent(in) :: s2
    real*8, intent(out) :: r

    r = sum(exp(-x**2/2/s2))
    
end subroutine gs