use super::*;

/// either type, TODO find in libraries
#[derive(Clone,Debug)]
pub enum Either<A,B> {
	Left(A),Right(B)
}

impl<T:Clone+PartialEq> Array3d<T>{
	fn iter_from_to(&self,start:V3i, end:V3i)->IterXYZ{
		IterXYZ::new(start,end, self.linear_index(start),self.linear_stride())
	}
	fn iter_at_sized(&self,start:V3i, size:V3i)->IterXYZ{
		IterXYZ::new(start,v3iadd(start,size), self.linear_index(start),self.linear_stride())
	}
	pub fn is_region_all_eq(&self,val:&T,start:V3i,end:V3i)->bool{
		for (pos,i) in self.iter_from_to(start,end){
			if self.at_linear_index(i)!=val{ return false}
		}
		return true;
	}
	pub fn is_region_homogeneous(&self,start:V3i,end:V3i)->Option<T>{
// grab 4 consecutive elements in the x axis
		let ref_val=self.index(start);
		if self.is_region_all_eq(ref_val,start,end){ Some(ref_val.clone())}
		else{None}
	}

	/// grab 4 consecutive elements in the x axis
	fn copy4_x(&self,pos:V3i)->[T;4]{
		assert!(pos.x+4<=self.shape.x);
		let i=self.linear_index(pos);
		// brute-force -construct the array immiediately-
		//avoids overwriting temporary
		[self.at_linear_index(i).clone(),self.at_linear_index(i+1).clone(),
		self.at_linear_index(i+2).clone(),self.at_linear_index(i+3).clone()]
	}
	fn copy4x4_xy(&self,pos:V3i)->[[T;4];4]{
		assert!(pos.y+4<=self.shape.y);
		[self.copy4_x(pos),self.copy4_x(v3iadd(pos,v3i(0,1,0))),
		self.copy4_x(v3iadd(pos,v3i(0,2,0))),self.copy4_x(v3iadd(pos,v3i(0,3,0)))]
	}

	fn copy4x4x4(&self,pos:V3i)->[[[T;4];4];4]{
		//setup and overwrite :(
		// to do cleanly+efficient.. need array4 constructor and repeated calls.
		assert!(pos.z+4<=self.shape.z);
		[self.copy4x4_xy(pos),self.copy4x4_xy(v3iadd(pos,v3i(0,0,1))),
		self.copy4x4_xy(v3iadd(pos,v3i(0,0,2))),self.copy4x4_xy(v3iadd(pos,v3i(0,0,3)))]
	}

	/// compressed tiling
	/// macro cells include either single value or detail
	/// 2x2x2 and 4x4x4 manually written.. awaiting <T,N>
	fn tile4(&self)->Array3d<Either<T,Box<[[[T;4];4];4]>>>{
		let cellsize=v3i(4,4,4);
		Array3d::from_fn(
			v3idiv(self.index_size(),cellsize),
			|pos|{
				let srcpos=v3imul(pos,cellsize);
				match self.is_region_homogeneous(srcpos,v3iadd(srcpos,cellsize)){
					Some(cell)=>Either::Left(cell),
					None=>Either::Right(Box::new(self.copy4x4x4(srcpos)))
				}
			}
		)
	}
}
type Tile4<T>=Either<T,Box<[[[T;4];4];4]>>;
pub struct Array3dTile4<T>(pub Array3d<Tile4<T>>);
impl<'a,T:Clone+PartialEq+Default> From<&'a Array3d<T>> for Array3dTile4<T>{
	fn from(s:&Array3d<T>)->Self{ Array3dTile4(s.tile4()) }
}
/// expand out raw array from, TODO - efficiently,
/// there should be a tiled constructor for Array3d<T> that works inplace
impl<'a,T:Clone> From<&'a Array3dTile4<T>> for Array3d<T> {
	fn from(src:&Array3dTile4<T>)->Array3d<T>{
		Array3d::from_fn(v3imuls(src.0.index_size(),4),|pos|{src[pos].clone()})
	}
}
/// read access to tile4 array
impl<T> Index<V3i> for Array3dTile4<T>{	
	type Output=T;
	fn index(&self,pos:V3i)->&T{
		let (tpos,sub)=v3itilepos(pos,2);
		match self.0.index(tpos){
			&Either::Left(ref x)=>&x,
			&Either::Right(ref tiledata)=>{
				&tiledata[sub.z as usize][sub.y as usize][sub.x as usize]
			}
		}
	}
}
fn clone4<T:Clone>(t:T)->[T;4]{
	[t.clone(),t.clone(),t.clone(),t]
}
/// write access to tile4 array TODO detect for writes that leave clear
impl<T:Clone> IndexMut<V3i> for Array3dTile4<T>{	
	fn index_mut(&mut self,pos:V3i)->&mut T{
		let (tpos,sub)=v3itilepos(pos,2);
		let a=self.0.index_mut(tpos);
		// convert the tile to mutable contents
		// unfortunately we can't erase yet
		// 2 steps to appease borrowck
		let newval=if let &mut Either::Left(ref val)=a{
			Some(val.clone())
		} else {None};
		if let Some(v)=newval{
			*a=Either::Right(Box::new(clone4(clone4(clone4(v)))));
		};
		// by now 'a' must be 'Right',i.e. a defined tile
		match *a{
			Either::Left(_)=>panic!("tile should be defined now"),
			Either::Right(ref mut p)=>&mut (*p)[sub.z as usize][sub.y as usize][sub.x as usize]
		}
		// after writes you must cleanup	
	}
}



