module density_f
    implicit none
    
contains
    
subroutine get_density_from_traj(density, dataset, feature_vars, coords, coeff, n_features)
    implicit none
    
    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_features
    real*8, intent(in) :: dataset(n_features), feature_vars(n_features), coords(n_features)
    real*8, intent(in) :: coeff    

    real*8 r2, temp
    integer n
    
    r2 = 0
    do n = 1, n_features
        temp = coords(n) - dataset(n)
        r2 = r2 + temp*temp/feature_vars(n)
    end do

    density = density + dexp(-coeff*r2)
end subroutine get_density_from_traj

subroutine get_density(density, dataset, feature_vars, coords, coeff, n_traj, n_features)
    implicit none
    
    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_traj, n_features
    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features), coords(n_features)
    real*8, intent(in) :: coeff

    integer i
    do i = 1, n_traj
        call get_density_from_traj(density, dataset(:,i), feature_vars, coords, coeff, n_features)
    end do
end subroutine get_density
    
end module