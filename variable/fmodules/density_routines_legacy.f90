subroutine get_density_from_traj_i4(density, dataset, feature_vars, use, coords, coeff, n_features, n_use)
    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_use, n_features
    real*8, intent(in) :: dataset(n_features), feature_vars(n_use)
    integer*4, intent(in) :: use(n_use), coords(n_use)
    real*8, intent(in) :: coeff    

    real*8 r2, temp
    integer n
    
    r2 = 0
    do n = 1, n_use
        temp = coords(n) - dataset(use(n))
        r2 = r2 + temp*temp/feature_vars(use(n))
    end do

    density = dexp(-coeff*r2)
end subroutine get_density_from_traj_i4

subroutine get_density_i4(density, dataset, feature_vars, use, coords, coeff, n_traj, n_features, n_use)
    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_traj, n_features, n_use
    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_use)
    integer*4, intent(in) ::  use(n_use), coords(n_use)
    real*8, intent(in) :: coeff

    integer i
    real*8 temp

    density = 0
    do i = 1, n_traj
        call get_density_from_traj_i4(temp, dataset(:,i), feature_vars, use, coords, coeff, n_features, n_use)
        density = density + temp
    end do
end subroutine get_density_i4

subroutine get_density_from_cluster_i4(density, dataset, feature_vars, use, coords, clusters, &
& coeff, clus_id, n_traj, n_features, n_use)
    real*8, intent(inout) :: density
!f2py intent(in,out) :: density
    
    integer*4, intent(in) :: n_traj, n_features, n_use, clus_id
    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_use)
    integer*4, intent(in) :: use(n_use), coords(n_use), clusters(n_traj)
    real*8, intent(in) :: coeff

    integer i

    density = 0
    do i = 1, n_traj
        if ((clusters(i) .eq. clus_id) .or. (clusters(i) .eq. clus_id - n_traj)) then
            call get_density_from_traj_i4(density, dataset(:,i), feature_vars, use, coords, coeff, n_features, n_use)
        end if
    end do
end subroutine get_density_from_cluster_i4