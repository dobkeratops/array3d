#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)] 
#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(slice_get_slice)]
#![feature(associated_type_defaults)]
#![feature(concat_idents)]
#![feature(step_trait)]

//! 2D,3D Array types (TODO-4d);
//! 
//! Includes a Tiled array type with simple compression, and a simpler array stored flat in memory.
//! 
//! Gives functions for internal & external iterators.
//! 
//! Indices use VecN .x .y .z types : this is for clarity
//! alongside code that uses array indices for acessing collections and tuple accessors for datastructures/multiple-return values.

#[cfg(test)]
mod tests {
	use super::*;
	#[test]
	fn iter_xyz_behaviour() {
		// test 1 axis of step
		let rng1=IterXYZ::new(v3i(1,2,3),v3i(4,3,4), 0,v3i(100,10,1));
		let cmp1=vec![(v3i(1,2,3),123), (v3i(2,2,3),223),(v3i(3,2,3),323)];

		let outp1:Vec<_>=rng1.collect(); // test iter-collect 
		assert_eq!(outp1,cmp1);

		// test x-z axes. 
		// use stride that places vlaues in digits for clarity
		let rng2=IterXYZ::new(v3i(1,2,3),v3i(3,3,5), 8000,v3i(1,10,100) );
		let cmp2=vec![(v3i(1,2,3),8321), (v3i(2,2,3),8322), (v3i(1,2,4),8421),(v3i(2,2,4),8422)];
		let mut outp2=Vec::new();
		for p in rng2 {	// test stepping through for notation
			outp2.push(p);
		}
		assert_eq!(outp2,cmp2);
		println!("foo\n");
	}

	fn my_array3d()->Array3d<i32>{
		Array3d::from_fn(v3i(2,2,1),|pos|{pos.x*100+pos.y*10+pos.z})
	}
	#[test]
	fn array3d_generation_and_indexing(){
		
		let foo=my_array3d();
		assert_eq!(foo[v3i(0,0,0)],000);
		assert_eq!(foo[v3i(0,1,0)],010);
		assert_eq!(foo[v3i(1,0,0)],100);
		assert_eq!(foo[v3i(1,1,0)],110);
	}
	#[test]
	fn array3d_iteration(){
		let foo=my_array3d();	
		
		let out:Vec<_>= foo.iter_cells().collect();
		let cmp=vec![(v3i(0,0,0),&000),(v3i(1,0,0),&100),(v3i(0,1,0),&010),(v3i(1,1,0),&110)];
		assert_eq!(out,cmp);
	}

	#[test]
	fn axis_fold(){
		let ar=Array3d::from_fn(v3i(2,2,2), |pos|pos.x*200+pos.y*30+pos.z*4);
		assert_eq!(ar.linear_index(v3i(1,1,1)),1+1*2+1*2*2);
		assert_eq!(ar[v3i(0,0,0)],000);
		assert_eq!(ar[v3i(1,0,0)],200);
		assert_eq!(ar[v3i(1,0,1)],204);
		assert_eq!(ar[v3i(0,0,1)],004);
		assert_eq!(ar[v3i(0,1,0)],030);
		assert_eq!(ar[v3i(0,1,1)],034);
		assert_eq!(ar[v3i(1,1,0)],230);
		assert_eq!(ar[v3i(1,1,1)],234);
		let ar2=ar.fold_z(0,|pos,a,b|a+b);
		assert_eq!(ar2[v2i(1,1)],464); // 230+234
		let ar3=ar.fold_x(0,|pos,a,b|a+b);
		assert_eq!(ar3[v2i(1,1)],268); // 034+234
	}
	#[test]
	fn array3d_iter_interface(){
		let ar=Array3d::from_fn(v3i(2,2,2), |pos|pos.x*200+pos.y*30+pos.z*4);
		let mut out:Vec<_>=Vec::new();
		for zs in ar.iter(){
			for ys in zs.iter(){
				for xs in ys.iter(){
					out.push(*xs);
				}
			}
		}
		assert_eq!(out[0],000);
		assert_eq!(out[1],200);
		assert_eq!(out[2],030);
		assert_eq!(out[3],230);

		let rng=v3i(0,0,0)..v3i(3,3,3);
	}
}


/// A Tiled 3d array with simple runtime-useable compression
pub mod tiled;
pub use tiled::*;
/// various maths utilities
pub mod math;
pub type Idx=i32;
use std::ops::Range;
use std::ops::{Add,Sub,Mul,Div,Rem,BitOr,BitAnd,BitXor,Index,IndexMut,Shl,Shr,AddAssign,SubAssign,MulAssign,DivAssign};
use std::cmp::PartialEq;
extern crate lininterp;
extern crate vec_xyzw;
use lininterp::{Lerp,avr};
fn div_rem(a:usize,b:usize)->(usize,usize){
	(a/b,a%b)
}

// local mini maths decouples
// TODO - make another crate declaring our style of Vec<T>
//#[derive(Copy,Debug,Clone)]
//pub struct Vec2<X,Y=X>{pub x:X,pub y:Y} // array3d::Vec3 should 'into()' into vector::Vec3 , etc.
//#[derive(Copy,Debug,Clone)]
//pub struct Vec3<X,Y=X,Z=Y>{pub x:X,pub y:Y,pub z:Z} // array3d::Vec3 should 'into()' into vector::Vec3 , etc.
//#[derive(Copy,Debug,Clone)]

pub use vec_xyzw::{Vec2,Vec3,Vec4,VElem};
pub use self::math::*;
//pub struct Vec4<X,Y=X,Z=Y,W=Z>{pub x:X,pub y:Y,pub z:Z,pub w:W} // array3d::Vec3 should 'into()' into vector::Vec3 , etc.
/// 2d index type allowing use of .x .y for clarity, compared to using arrays for indices and the collection itself, or tuples. 'i'=32bit signed integer
pub type V2i=Vec2<Idx>;
/// 3d index type allowing use of .x .y,.z for clarity, compared to using arrays for indices and the collection itself, or tuples. 'i'=32bit signed integer
pub type V3i=Vec3<Idx>;
/// 4d index type allowing use of .x .y,.z,.w for clarity, compared to using arrays for indices and the collection itself, or tuples. 'i'=32bit signed integer
pub type V4i=Vec4<Idx>;//TODO
type Axis_t=i32;

trait GetMut<I>{
	type Output;
	fn get_mut(&mut self, i:I)->&mut Self::Output;
}

impl<T:VElem> GetMut<i32> for Vec3<T>{
	type Output=T;
	fn get_mut(&mut self, i:i32)->&mut T{
		match i{
			0=>&mut self.x, 1=>&mut self.y, 2=>&mut self.z,
			_=>panic!("axis index out of range")
		}		
	}
}

const XAxis:Axis_t=0; const YAxis:Axis_t=1; const ZAxis:Axis_t=2;
/*
impl<T> Index<i32> for Vec3<T>{
	type Output=T;
	fn index(&self,i:Axis_t)->&T{match i{ XAxis=>&self.x,YAxis=>&self.y,ZAxis=>&self.z,_=>panic!("Vec3 index out of range")}}
}
impl<T> IndexMut<i32> for Ve3<T>{
	fn index_mut(&mut self,i:Axis_t)->&mut T{match i{ XAxis=>&mut self.x,YAxis=>&mut self.y,ZAxis=>&mut self.z,_=>panic!("Vec3 index out of range")}}
}
*/
pub fn v2i(x:i32,y:i32)->V2i{Vec2{x:x,y:y}}
pub fn v3i(x:i32,y:i32,z:i32)->V3i{Vec3{x:x,y:y,z:z}}
pub fn v3izero()->V3i{v3i(0,0,0)}
pub fn v3ione()->V3i{v3i(1,1,1)}
pub fn v3iadd_axis(a:V3i,axis:i32, value:i32)->V3i{let mut r=a; *r.get_mut(axis)+=value; r}

pub fn v3f_index_fraction(pos:Vec3<f32>)->(V3i,Vec3<f32>){
	assert!(pos.x>=0.0&&pos.y>=0.0&&pos.z>=0.0);
//	let |f|{assert!(f>=0.0);if f<0.0{(f.ceil(),1.0-f.fract())}else{(f.floor(),f.fract())}}; 
	// todo-vectorizable .. indices processed in simd reg.
	let ipos=v3i(pos.x.floor() as i32,pos.y.floor() as i32,pos.z.floor() as i32);
	let fpos=Vec3{x:pos.x.fract(),y:pos.y.fract(),z:pos.z.fract()};

	(ipos,fpos)	
}

/// Stores neighbours in one axis for a cell
#[derive(Debug,Copy,Clone)]
pub struct Neighbours<T>{pub prev:T,pub next:T}
/// neighbours for a 2d array cell
pub type Neighbours2d<T>=Vec2<Neighbours<T>>;
/// neighbours for a 3d array cell
pub type Neighbours3d<T>=Vec3<Neighbours<T>>;
/// neighbours for a 4d array cell
pub type Neighbours4d<T>=Vec4<Neighbours<T>>;

/// Flat 2d array, stored linearly in a single Vec<T>
pub struct Array2d<T>{pub shape:V2i,pub data:Vec<T>}
/// Flat 3d array, stored linearly in a single Vec<T>
pub struct Array3d<T>{pub shape:V3i,pub data:Vec<T>}
/// Flat 4d array, stored linearly in a single Vec<T>
pub struct Array4d<T>{pub shape:V4i,pub data:Vec<T>}

fn eval_linear_index(pos:V3i, stride:V3i)->usize{
	(pos.x as usize)*(stride.x as usize)+(pos.y as usize) * (stride.y as usize) + (pos.z as usize)*(stride.z as usize)
}

/// iteration state through XY planes, has current Z
pub struct SliceZIter<'a,T:'a>{
	pub z:i32,zmax:i32,
	pub data:&'a Array3d<T>
}
pub struct SliceZYIter<'a,T:'a>{
	pub z:i32,y:i32,ymax:i32,
	pub data:&'a Array3d<T>
}
/// an XY slice, at a given Z
pub struct SliceZ<'a,T:'a>{
	pub z:i32, pub data:&'a Array3d<T>
}
pub struct SliceZY<'a,T:'a>{
	pub z:i32,y:i32,pub data:&'a Array3d<T>
}
impl<'a,T:'a> SliceZY<'a, T>{
	fn iter(&self)->SliceZYXIter<'a,T>{
		SliceZYXIter{
			z:self.z,y:self.y,x:0 ,xmax:self.data.index_size().x, data:self.data
		}
	}
}
impl<'a,T:'a> SliceZ<'a, T>{
	fn iter(&self)->SliceZYIter<'a,T>{
		SliceZYIter{
			z:self.z,y:0,ymax:self.data.index_size().y,data:self.data
		}
	}
}

impl<'a,T:'a> Iterator for SliceZIter<'a,T>{
	type Item=SliceZ<'a,T>;
	fn next(&mut self)->Option<SliceZ<'a,T>>{
		if self.z>=self.zmax{
			return None;
		} else{
			let ret=SliceZ{z:self.z,data:self.data};
			self.z+=1;
			return Some(ret);
		}
	}
}

/// iterator for X slices (at given Z,Y)-
impl<'a,T:'a> Iterator for SliceZYIter<'a,T>
{
	type Item=SliceZY<'a,T>;
	fn next(&mut self)->Option<SliceZY<'a,T>>{
		if self.y>=self.ymax{
			return None;
		} else {
			let ret=SliceZY{
				z:self.z,y:self.y,data:self.data
			};
			self.y+=1;
			return Some(ret);
		}
	}
}

struct SliceZYXIter<'a,T:'a>{
	x:i32,y:i32,z:i32,xmax:i32,data:&'a Array3d<T>
}

struct SliceZYX<'a,T:'a>{
	x:i32,y:i32,z:i32,data:&'a Array3d<T>	
}

impl<'a,T:'a> Iterator for SliceZYXIter<'a,T>
{
	// todo: should this actually return an
	//intermediate SliceZYX which still carries x/y/z
	// & derefs to T?
	type Item=&'a T;
	fn next(&mut self)->Option<Self::Item> {
		if self.x>=self.xmax{
			return None;
		} else {
			let ret=&self.data[v3i(self.x,self.y,self.z)];
			self.x+=1;
			return Some(ret);
		}
	}
}


/// Utility trait: any collections which are indexable by XYZ
pub trait XYZIndexable : IndexMut<V3i>{
	fn index_size(&self)->V3i;
}


/// utility trait: any collections which can convert between XYZ and single linear indices. Will allow more efficient impls of traversals
pub trait XYZLinearIndexable : XYZIndexable{
	fn linear_index(&self,pos:V3i)->LinearIndex;
	fn pos_from_linear_index(&self,LinearIndex)->V3i;	
}



/*TODO , what is rust assignment operator?
impl<'a,'b,'c, T:Clone> Assign<Array3dSlice<'a,T>> for Array3dSlice<'b,T>{
}
*/
impl<T:Clone> Array2d<T>{
	pub fn new()->Self{Array2d{shape:v2i(0,0),data:Vec::new()}}
	pub fn len(&self)->usize{ v2i_hmul_usize(self.shape) }
	pub fn linear_index(&self, pos:V2i)->usize{
		// now this *could* exceed 2gb.
		(pos.x as usize)+
		(self.shape.x as usize) *(pos.y as usize)
	}
	pub fn map_pos<B:Clone,F:Fn(V2i,&T)->B> (&self,f:F) -> Array2d<B>{
		// todo xyz iterator
		let mut r=Array2d::new();
		r.data.reserve(self.data.len());
		
		for y in 0..self.shape.y { for x in 0.. self.shape.x{
			let pos=v2i(x,y);
			r.data.push( f(pos,self.index(pos)) );
		}}
		r
	}

	pub fn from_fn<F:Fn(V2i)->T> (s:V2i, f:F)->Array2d<T>{
		let mut d=Array2d{shape:s,data:Vec::new()};
		d.data.reserve(s.x as usize * s.y as usize);
		for y in 0..s.y{ for x in 0..s.x{
				d.data.push(f(v2i(x,y)))
			}
		}
		d
	}
	pub fn from_fn_linear_indexed<F:Fn(V2i,LinearIndex)->T> (s:V2i, f:F)->Array2d<T>{
		let mut d=Array2d{shape:s,data:Vec::new()};
		d.data.reserve(s.x as usize * s.y as usize);
		let mut i=0;
		for y in 0..s.y{ for x in 0..s.x{
				d.data.push(f(v2i(x,y),i)); i+=1;
			}
		}
		d
	}
}

impl<T:Clone> Index<V2i> for Array2d<T>{
	type Output=T;
	fn index(&self, pos:V2i)->&T{
		let i=self.linear_index(pos);
		&self.data[i]
		
	}
}

impl<T:Clone> IndexMut<V2i> for Array2d<T>{
	fn index_mut(&mut self, pos:V2i)->&mut T{
		let i=self.linear_index(pos);
		&mut self.data[i]
	}
}

/// Helper for iterations storing index range and some precomputed linear indices
pub struct IterXYZRange {
	start:V3i,
	end:V3i,
	step:V3i,
	li_base:LinearIndex,
	li_stride:V3i,
	li_step:usize,
}
type LinearIndex=usize;

/// state for an XYZ iteration with linear index,
/// todo- seperate into pure xyz 
/// and linear index adapter
pub struct IterXYZState{
	pos:V3i,
	linear_index:LinearIndex,	// array index
}
/// Iterator over 3d index positions,maintaining an associated linear index computed from strides
pub struct IterXYZ {
	state:IterXYZState,
	range:IterXYZRange
}

/// for output of <IterXYZ as Iterator> - 3d index and associated linear index
pub type IterXYZItem=(V3i,usize);
/// helper function: perform a step of XYZ range iteration - updates an IterXYZState, assuming it's controlled by the given range.
pub fn step_xyz_iter(s:&mut IterXYZState,range:&IterXYZRange)->Option<IterXYZItem>{
// todo - evaluate 'INDEX'incrementally, thats the point!
	if s.pos.z>=range.end.z {return None;}
	let ret=(s.pos,s.linear_index);
	s.pos.x+=range.step.x;
	s.linear_index += range.li_step;
	if s.pos.x>=range.end.x{
		s.pos.x=range.start.x;
		s.pos.y+=range.step.y;
		if s.pos.y>=range.end.y{
			s.pos.y=range.start.y;
			s.pos.z+=range.step.z;
		}
		s.linear_index=range.eval_linear_index(s.pos);
	}
	Some(ret)
}
/// Part of external iterator for XYZ values - contains the unchanging description of the whole iteration
impl IterXYZRange{
	fn eval_linear_index(&self,pos:V3i)->usize{
		(pos.x as usize*self.li_stride.x as usize)+
		(pos.y as usize*self.li_stride.y as usize)+
		(pos.z as usize*self.li_stride.z as usize)+
		self.li_base
	}
}

impl Iterator for IterXYZ{
	type Item=IterXYZItem;
	fn next(&mut self)->Option<Self::Item>{
		step_xyz_iter(&mut self.state,&self.range)
	}
}


/// range-bounded Region of array3d, analogous to array slices.
/// 'TS' type-param =collecton of T's, T used for Items
pub struct RangeOf<'a,I,TS:'a+Index<I>>(Range<I>, &'a TS );
pub struct RangeOfMut<'a,I,TS:'a+IndexMut<I>+Index<I>>(Range<I>, &'a mut TS );

impl<'a,I,TS:'a+Index<I>> Index<I> for RangeOf<'a,I,TS> {
	type Output=<TS as Index<I>>::Output;
	fn index(&self,i:I)->&Self::Output{
		self.1.index(i)
	}
}
impl<'a,I,TS:'a+IndexMut<I>> Index<I> for RangeOfMut<'a,I,TS> {
	type Output=<TS as Index<I>>::Output;
	fn index(&self,i:I)->&Self::Output{
		self.1.index(i)
	}
}
impl<'a,I,TS:'a+IndexMut<I>> IndexMut<I> for RangeOfMut<'a,I,TS> {
	fn index_mut(&mut self,i:I)->&mut Self::Output{
		self.1.index_mut(i)
	}
}


impl IterXYZ{
	fn new(start:V3i, end:V3i, base_index:usize , index_stride:V3i)->IterXYZ
	{
		let rng=IterXYZRange{
				li_base:base_index,	
				li_stride:index_stride,
				li_step:1 * index_stride.x as usize,
				start:start,
				end:end,
				step:v3i(1,1,1)
			};
		IterXYZ{
			state:IterXYZState{
				linear_index: rng.eval_linear_index(start),
				pos:start,
			},
			range:rng,
		}	
	}
	// 'step' assigned in builder-like manner similar to itertools
	fn step(self,step:V3i)->IterXYZ{
		let mut r=self;
		r.range.step=step;
		r.range.li_step=r.range.step.x as usize* r.range.li_stride.x as usize;
		r
	}
}

/// interfacing generic 3d array stores/views with iterators
/// uses this interface, supporting 3d indices&linear indices
/// e.g. in a copy allows traversing one linearly

pub trait IterXYZAble<'a> : Index<V3i> + XYZLinearIndexable{
	type Elem;
	fn at_linear_index(&'a self,i:usize)->&'a Self::Elem;
}
pub trait IterXYZAbleMut<'a> : IndexMut<V3i> + IterXYZAble<'a>
{
	fn at_linear_index_mut(&'a mut self, i:usize)->&'a mut Self::Elem;
}


/*
TODO we can't figure out the lifetimes/borrowchecker markup for this
this should have let us share impl for slice & vec iteration
Vec iteration was working, implemented directly.
/// iterating an 'array3d' behaves like .iter().enumerate()
/// todo - should we actually give slice2ds, slices?

/// pair combining a 3d iteration description with a collection it's in
struct IterXYZIn<'a,  A:IterXYZAble<'a>+'a>(&'a A, IterXYZ);	
struct IterXYZInMut<'a, A:IterXYZAble<'a>+'a>(&'a mut A, IterXYZ);	

impl<'a, A:IterXYZAble<'a>> Iterator for IterXYZIn<'a,A> {
	type Item=(V3i,&'a A::Elem);
	fn next(&mut self)->Option<Self::Item>{
		if let Some(curr)=self.1.next(){
			Some((curr.0, self.0.at_linear_index(curr.1)))
		} else {None}
	}
}
impl<'a,A:IterXYZAble<'a>> Iterator for IterXYZInMut<'a,A>{
	type Item=(V3i,&'a mut A::Elem);
	fn next(&mut self)->Option<Self::Item>{
		if let Some(curr)=self.1.next(){
			let elem:&'a mut A::Elem = (&self).0.at_linear_index_mut(curr.1);
			Some((curr.0, elem ))
		} else {None}
	}
}
*/

impl<T:Clone> Array3d<T>{
	/// consume a vec to build. opposite of to_vec
	fn from_vec(size:V3i,v:Vec<T>)->Self{
		assert!((size.x*size.y*size.z) as usize==v.len());
		Array3d{shape:size,data:v}
	}
	/// destructure into the shape and flat array seperately; inverse of from_vec
	fn to_vec(self)->(V3i,Vec<T>){
		(self.shape,self.data)
	}
	/// get a region, TODO can this be made to 
	/// work with rust's nifty array syntax? Index<.> seems to require returning &Output, which precludes returning a wrapper helper-strucure
	fn range<'a>(&'a self, rng:Range<V3i>)->RangeOf<'a,V3i,Self>{
		RangeOf(rng,self)		
	}

	/// keep the same underlying sequential buffer memory,but re-interpret as a different array shape
	fn reshape(&mut self, s:V3i){
		// todo: do allow truncations etc
		// todo: reshape to 2d, 3d
		assert!(v3i_hmul_usize(self.shape)==v3i_hmul_usize(s));
		self.shape=s;
	}

}

/// struct holding iteration info for a 3d array
struct Array3dIter<'a,T:'a>(&'a Array3d<T>,IterXYZ);

/// struct holding iteration info for a 3d array
struct Array3dIterMut<'a,T:'a>(&'a mut Array3d<T>,IterXYZ);

impl<'a, T:Clone+'a> Iterator for Array3dIter<'a,T> {
	type Item=(V3i,&'a T);
	fn next(&mut self)->Option<Self::Item>{
		if let Some(curr)=self.1.next(){
			Some((curr.0, self.0.at_linear_index(curr.1)))
		} else {None}
	}
}/*
impl<'a, T:Clone+'a> Iterator for Array3dIterMut<'a,T> {
	type Item=(V3i,&'a mut T);
	fn next(&mut self)->Option<Self::Item>{
		if let Some(curr)=self.1.next(){
			Some((curr.0, self.0.at_linear_index_mut(curr.1)))
		} else {None}
	}
}
*/

impl<T:Clone> Array3d<T>{// TODO is there an 'iterable' trait?

	/// iterator returning representation of 2d xy slices at a given z; Allows Array3d to be iterated in a manner similar to Vec<Vec<Vec<T>>>. See also 'iter_cells' for contrasting unified x/y/z enumeration
	fn iter<'a>(&'a self)->SliceZIter<'a,T>{
		SliceZIter{z:0,zmax:self.index_size().z, data: self}
	}
	/// iterate through the entire 3d array, passing the cell reference along with it's 3d index; like a 3d version of iter().enumerate()
	fn iter_cells<'a>(&'a self)->Array3dIter<'a,T>{
		Array3dIter(self, IterXYZ::new(v3i(0,0,0),self.shape, 0 , self.linear_stride()))
	}
	/// mutable version of iter_cells
	fn iter_cells_mut<'a>(&'a mut self)->Array3dIterMut<'a,T>{
		let shp=self.shape.clone();
		let strd=self.linear_stride();
		Array3dIterMut(self, IterXYZ::new(v3i(0,0,0),shp, 0 , strd))
	}
	/// compute the index scales of x/y/z respectively
	/// x-stride assumed not necaserily one, to allow
	/// swizzled/strided views etc.
	fn linear_stride(&self)->V3i{v3i(1,self.shape.x,self.shape.x*self.shape.y)}
}

impl<T> XYZIndexable for Array3d<T>{
	fn index_size(&self)->V3i{self.shape}
}
impl<T> XYZLinearIndexable for Array3d<T>{
	fn linear_index(&self, pos:V3i)->LinearIndex{
		let shape=self.index_size();
		// now this *could* exceed 2gb.
		(pos.x as usize)+
		(shape.x as usize)*( 
			(pos.y as usize)+
			(pos.z as usize)*(shape.y as usize)
		)
	}
	fn pos_from_linear_index(&self,i:LinearIndex)->V3i{
		unimplemented!()
//		let (x,rest)=
	}
}

impl<'a,T> IterXYZAble<'a> for Array3d<T>{
	type Elem=T;
	fn at_linear_index(&'a self,i:LinearIndex)->&'a T{ self.data.index(i)}
}
impl<'a,T> IterXYZAbleMut<'a> for Array3d<T>{
	fn at_linear_index_mut(&'a mut self,i:LinearIndex)->&'a mut T{ self.data.index_mut(i)}
}

impl<T:Clone> Array3d<T>{

	/// produce an array from a function applied to cell indices	
	pub fn from_fn<F:Fn(V3i)->T> (s:V3i,f:F) -> Array3d<T> {
		let mut a=Array3d{shape:s, data:Vec::new()};
		a.data.reserve(v3i_hmul_usize(s));
		for z in 0..a.shape.z{ for y in 0..a.shape.y { for x in 0.. a.shape.x{
			a.data.push( f(v3i(x,y,z)) )
		}}}
		a
	}

	/// produce an array from a function applied to cell indices, with the linear index available
	pub fn from_fn_linear_indexed<F:Fn(V3i,i32)->T> (s:V3i,f:F) -> Array3d<T> {
		let mut a=Array3d{shape:s, data:Vec::new()};
		a.data.reserve(v3i_hmul_usize(s));
		let mut i=0;
		for z in 0..a.shape.z{ for y in 0..a.shape.y { for x in 0.. a.shape.x{
			a.data.push( f(v3i(x,y,z),i) );
			i+=1;
		}}}
		a
	}

	/// production from a function with expansion,
	/// todo - make this more general e.g.'production in blocks'
	/// the motivation is to share state across the production of
	///  a block of adjacent cells
	pub fn from_fn_doubled<F:Fn(V3i)->(T,T)> (sz:V3i,axis:i32, f:F)->Array3d<T>{
		let mut scale=v3i(1,1,1); *scale.get_mut(axis)*=2;
		let mut d=Array3d{shape:v3imul(sz,scale),data:Vec::new()};
		d.data.reserve(sz.x as usize * sz.y as usize * sz.z as usize);
		for z in 0..sz.z {for y in 0..sz.y{ for x in 0..sz.x{
					let (v0,v1)=f(v3i(x,y,z));
					d.data.push(v0);
					d.data.push(v1);
				}
			}
		}
		d
	}

	pub fn fill_val(size:V3i,val:&T)->Array3d<T>{let pval=&val;Self::from_fn(size,|_|val.clone())}

	/// initialize with repeat value:
	/// synonym follows naming convention 'from_'
	pub fn from_val(size:V3i,val:&T)->Self{Self::fill_val(size,val)}

	pub fn len(&self)->usize{ v3i_hmul_usize(self.shape) }
}

impl<TS:XYZIndexable> XYZInternalIterators for TS{}

macro_rules! fold_axis{(fn $fname:ident traverse ($u:ident,$v:ident) reduce $w:ident )=>{
	/// produce a 2d array by folding along the specified axis, with the resulting axes corresponding to the others
	fn $fname<B:Clone,F:Fn(V3i,B,&Self::Output)->B> (&self,input:B, f:F) -> Array2d<B>{
		let mut out=Array2d::new();
		let shape=self.index_size();
		out.data.reserve(shape.$u as usize *shape.$v as usize);
		let mut pos=v3i(0,0,0);//yuk. for macro
		pos.$v=0;
		for $v in 0..shape.$v {
			pos.$u=0;
			for $u in 0.. shape.$u{
				let mut acc=input.clone();
				pos.$w=0;
				for $w in 0..shape.$w{
//					let pos=v3i(x,y,z);
					acc=f(pos,acc, self.index(pos));
					pos.$w += 1;
				}
				out.data.push(acc);
				pos.$u+=1;
			}
			pos.$v+=1;
		}		
		out.shape=v2i(shape.$u, shape.$v);
		out		
	}
}}

impl<T:lininterp::Lerp+Clone> Array3d<T>{
	/// perform trilinear sampled read of cell data
	/// assumes the fracitional part is the interpolant
	fn trilinear_sample(&self,pos:Vec3<f32>)->T{
		assert!(pos.x>=0.0 && pos.y>=0.0 && pos.z>=0.0,"negative values not handled yet");
		let (ipos,fpos)=v3f_index_fraction(pos);
		lininterp::trilerp(&self.get2x2x2(ipos),(fpos.x,fpos.y,fpos.z))
	}
}


/// Internal iterators, and some associated helper functions;
///
/// Until rust has HKT/ATOC, options for the output are limited:
/// we just output a flat Array3d for the moment.
/// Default implementations are just built on indexing; 
/// however cell address-generation will be inefficient this way.
///
/// Implementation for specific arrangements could implement certain
/// traversals more efficiently (be it flat, morton order, whatever)
///
/// TODO - try to re-arrange default around a 'scanline' (X-slice) helper object to at least get reasonably efficient Flat array use out of the default impl.
/// TODO - data-parallel versions.
pub trait XYZInternalIterators : XYZIndexable{
	// TODO - looks gross using Self::Output instead of 'T' as per impls,
	// is there a way to make a local alias? always seems to need 'Self::..'
	// output is 'Elem' , happens to come from Index<T> .
	// causes confusion with *output* for individual functions here.
	/// get a 2x2x2 region of the array-adjacent values eg for filtering/convolution
	fn get2x2x2(&self,pos:V3i)->[[[&Self::Output;2];2];2]{
		[self.get2x2(pos),self.get2x2(pos+v3i(0,0,1))]
	}
	fn get2(&self,pos:V3i)->[&Self::Output;2]{
		[self.index(pos),self.index(pos+v3i(1,0,0))]
	}
	fn get2x2(&self,pos:V3i)->[[&Self::Output;2];2]{
		[self.get2(pos),self.get2(pos+v3i(0,1,0))]
	}

	fn map_strided_region<B:Clone,F:Fn(V3i,&Self::Output)->B> 
		(&self,range:Range<V3i>,stride:V3i, f:F) -> Array3d<B>
	{
		Array3d::from_fn((range.end-range.start)/stride,
			|outpos:V3i|{
				let inpos=v3iadd(outpos*stride,range.start);
				f(inpos,self.index(inpos))
			}
		)
	}

	/// zip_with - apply a function to corresponding elements from a pair of arrays to produce a new array of the results
	fn zip_with<B,R,F>(&self,other:&B,f:F)->Array3d<R>
		where B:XYZInternalIterators, F:Fn(V3i, &Self::Output, &B::Output)->R, R:Clone {
		assert!(self.index_size()==other.index_size());
		Array3d::from_fn(self.index_size(), |pos:V3i|f(pos,self.index(pos),other.index(pos)))
//		Array3d::from_fn_linear_indexed(self.index_size(), |pos:V3i,i:i32|f(self.at_linear_index[i],other.at_linear_index[i]) })
	}

	/// produce a new array by applying a function to each element
	fn map_xyz<B:Clone,F:Fn(V3i,&Self::Output)->B> (&self,f:F) -> Array3d<B>{
		Array3d::from_fn(self.index_size(),
			|pos:V3i|f(pos,self.index(pos)))
	}
	/// internal iteration with inplace mutation
	fn for_each<F>(&mut self,f:F) 
		where F:Fn(V3i,&mut Self::Output)
	{
		let shape=self.index_size();
		for z in 0..shape.z{ for y in 0..shape.y { for x in 0.. shape.x{
			let pos=v3i(x,y,z);
			f(pos,self.index_mut(pos))
		}}}
	}

	// mappers along each pair of axes,
	// form primitive for reducers along the axes
	// or slice extraction
	fn map_xy<F,B>(&self, a_func:F)->Array2d<B>
		where F:Fn(&Self,i32,i32)->B, B:Clone
	{
		let sz=self.index_size();
		Array2d::from_fn(v2i(sz.x,sz.z),
			|pos:V2i|{a_func(self,pos.x,pos.y)}
		)		
	}
	fn map_xz<F,B>(&self, a_func:F)->Array2d<B>
		where F:Fn(&Self,i32,i32)->B, B:Clone
	{
		let sz=self.index_size();
		Array2d::from_fn(v2i(sz.x,sz.z),
			|pos:V2i|{a_func(self,pos.x,pos.y)}
		)		
	}
	fn map_yz<F,B>(&self, a_func:F)->Array2d<B>
		where F:Fn(&Self,i32,i32)->B, B:Clone
	{
		let sz=self.index_size();
		Array2d::from_fn(v2i(sz.y, sz.z),
			|pos:V2i|{a_func(self,pos.x,pos.y)}
		)		
	}


	// TODO ability to do the opposite e.g. map to vec's which become extra dim.
	// fold along axes collapse the array ?
	// e.g. run length encoding,packing, whatever.
/*
	fn fold_z<B,F> (&self,init_val:B, f:F) -> Array2d<B>
		where B:Clone,F:Fn(V3i,B,&Self::Output)->B
	{
		self.map_xy(|s:&Self,x:i32,y:i32|{
			let s_shape=s.index_size();
			let mut acc=init_val.clone();
			for z in 0..s_shape.z{
				let pos=v3i(x,y,z);
				acc=f(pos,acc,self.index(pos))
			}
			acc
		})
	}
*/
/*
		let mut out=Array2d::new();
		out.data.reserve(self.shape.y as usize *self.shape.z as usize);
		for y in 0..self.shape.y { for x in 0.. self.shape.x{
			let mut acc=input.clone();
			for z in 0..self.shape.z{
				let pos=Vec3(x,y,z);
				acc=f(pos,acc,self.index(pos));
			}
			out.data.push(acc);
		}}		
		out.shape=Vec2(self.shape.x,self.shape.y);
		out
	}
*/
	/// fold values along the x axis, producing a 2d array corresponding to yz
	fold_axis!{fn fold_x traverse (y,z) reduce x}
	/// fold values along the y axis, producing a 2d array corresponding to xz
	fold_axis!{fn fold_y traverse (x,z) reduce y}
	/// fold values along the z axis, producing a 2d array corresponding to xy
	fold_axis!{fn fold_z traverse (x,y) reduce z}
/*
	/// produce a 2d array by folding along the X axis
	fn fold_x<B:Clone,F:Fn(V3i,B,&Self::Output)->B> (&self,input:B, f:F) -> Array2d<B>{
		let mut out=Array2d::new();
		let shape=self.index_size();
		out.data.reserve(shape.y as usize *shape.z as usize);
		for z in 0..shape.z {
			for y in 0.. shape.y{
				let mut acc=input.clone();
				for x in 0..shape.x{
					let pos=v3i(x,y,z);
					acc=f(pos,acc, self.index(pos));
				}
				out.data.push(acc);
			}
		}		
		out.shape=v2i(shape.y, shape.z);
		out		
	}
*/
	/// fold values along z,y,x axes to a single result without intermediate storage
	fn fold_xyz<A,B,C,FOLDX,FOLDY,FOLDZ>(
		&self,
		ifx:(A,FOLDX), ify:(B,FOLDY), ifz:(C,FOLDZ)
	)->A
	where
				A:Clone,B:Clone,C:Clone,
				FOLDZ:Fn(V3i,C,&Self::Output)->C,
				FOLDY:Fn(i32,i32,B,&C)->B,
				FOLDX:Fn(i32,A,&B)->A,
	{
		let (input_x,fx)=ifx;
		let (input_y,fy)=ify;
		let (input_z,fz)=ifz;
		let mut ax=input_x.clone();
		let shape=self.index_size();
		for x in 0..shape.x{
			let mut ay=input_y.clone();
			for y in 0..shape.y{
				let mut az=input_z.clone();//x accumulator
				for z in 0..shape.z{
					let pos=v3i(x,y,z);
					az=fz(pos,az,self.index(pos));
				}
				ay=fy(x,y,ay,&az);
			}
			ax= fx(x, ax,&ay);
		}
		ax
	}
	/// fold values along x,y,z in turn without intermediate storage
	fn fold_zyx<A,B,C,FOLDX,FOLDY,FOLDZ>(
		&self,
		ifx:(A,FOLDX), ify:(B,FOLDY), ifz:(C,FOLDZ)
	)->C
	where
				A:Clone,B:Clone,C:Clone,
				FOLDZ:Fn(i32,C,&B)->C,
				FOLDY:Fn(i32,i32,B,&A)->B,
				FOLDX:Fn(V3i,A,&Self::Output)->A,
	{
		let (input_x,fx)=ifx;
		let (input_y,fy)=ify;
		let (input_z,fz)=ifz;
	
		let shape=self.index_size();
		let mut az=input_z.clone();
		for z in 0.. shape.z{
			let mut ay=input_y.clone();
			for y in 0..shape.y{
				let mut ax=input_x.clone();//x accumulator
				for x in 0..shape.x{
					let pos=v3i(x,y,z);
					ax=fx(pos,ax,self.index(pos));
				}
				ay=fy(y,z,ay,&ax);
			}
			az= fz(z, az,&ay);
		}
		az
	}

	/// fold the whole array to produce a single value
	fn fold<B,F> (&self,input:B, f:F) -> B
	where F:Fn(V3i,B,&Self::Output)->B,B:Clone
	{
		let mut acc=input;
		let shape=self.index_size();
		for z in 0..shape.z { for y in 0.. shape.y{ for x in 0..shape.x{
			let pos=v3i(x,y,z);
			acc=f(pos, acc,self.index(pos));
		}}}
		acc
	}

	/// produce tiles by applying a function to every subtile
	///
	/// The output size is divided by the tilesize;
	/// there must be an exact multiple of the given tilesize in the inputs 
	fn fold_tiles<B,F>(&self,tilesize:V3i, input:B,f:&F)->Array3d<B>
		where F:Fn(V3i,B,&Self::Output)->B,B:Clone
	{
		assert!(self.index_size()==(self.index_size()/tilesize)*tilesize);
		self.map_strided(tilesize,
			|pos,_:&Self::Output|{self.fold_region(pos..(pos+tilesize),input.clone(),f)})
	}

	/// subroutine for 'fold tiles', see context
	/// closure is borrowed for multiple invocation by caller
	fn fold_region<B,F>(&self,r:Range<V3i>, input:B,f:&F)->B
		where F:Fn(V3i,B,&Self::Output)->B, B:Clone
	{
		let mut acc=input.clone();
		for z in r.start.z..r.end.z{
			for y in r.start.y..r.end.y{
				for x in r.start.x..r.end.x{
					let pos=v3i(x,y,z);
					acc=f(pos,acc,self.index(pos));
				}
			}
		}
		acc
	}
	fn get_indexed(&self,pos:V3i)->(V3i,&Self::Output){(pos,self.index(pos))}
	fn region_all(&self)->Range<V3i>{v3i(0,0,0)..self.index_size()}
	fn map_region_strided<F,B>(&self,region:Range<V3i>,stride:V3i, f:F)->Array3d<B>
		where F:Fn(V3i,&Self::Output)->B, B:Clone{
		Array3d::from_fn((region.end-region.start)/stride,
			|outpos:V3i|{
				let inpos=region.start+outpos*stride;
				f(inpos,self.index(inpos))  
			}
		)
	}
	fn map_strided<F,B>(&self,stride:V3i,f:F)->Array3d<B>
		where F:Fn(V3i,&Self::Output)->B, B:Clone{
		self.map_region_strided(self.region_all(),stride,f)
	}
	fn map_region<F,B>(&self,region:Range<V3i>,f:F)->Array3d<B>
		where F:Fn(V3i,&Self::Output)->B, B:Clone{
		self.map_region_strided(region, v3i(1,1,1), f)
	}
	/// Simple convolution which only considers a cell with it's immediate neighbours along each axis, i.e. inputs are (x,y,z), (x-1,y,z),(x+1,y,z), (x,x-1,z),(x,x+1,z), (x,y,z-1),(x,y,z+1)
	fn convolute_neighbours<F,B>(&self,f:F)->Array3d<B>
		where F:Fn(&Self::Output,Vec3<Neighbours<&Self::Output>>)->B ,B:Clone
	{
		self.map_region(v3i(1,1,1)..(self.index_size()-v3i(1,1,1)),
			|pos:V3i,current_cell:&Self::Output|{
				f(	current_cell,
					self::Vec3{
						x:Neighbours{
							prev:self.index(pos+v3i(-1,0,0)),
							next:self.index(pos+v3i(1,0,0)) },
						y:Neighbours{
							prev:self.index(pos+v3i(0,-1,0)),
							next:self.index(pos+v3i(0,1,0)) },
						z:Neighbours{
							prev:self.index(pos+v3i(0,0,-1)),
							next:self.index(pos+v3i(0,0,1)) }})
		})
	}
	/// version of indexed element access allowing wrap-around indices
	fn index_wrap(&self,pos:V3i)->&Self::Output{self.get_wrap(pos)}
	fn get_wrap(&self,pos:V3i)->&Self::Output{
		self.index( v3imymod(pos, self.index_size()) )
	}
	fn get_ofs_wrap(&self,pos:V3i,dx:i32,dy:i32,dz:i32)->&Self::Output{
		self.get_wrap(pos+ v3i(dx,dy,dz))
	}
	fn convolute_neighbours_wrap<F,B>(&self,f:F)->Array3d<B>
		where F:Fn(&Self::Output,Vec3<Neighbours<&Self::Output>>)->B,B:Clone 
	{
		// TODO - efficiently, i.e. share the offset addresses internally
		// and compute the edges explicitely
		// niave implementation calls mod for x/y/z individually and forms address
		let shape=self.index_size();
		self.map_region(v3izero()..shape,
			|pos:V3i,current_cell:&Self::Output|{
				f(	current_cell,
					Vec3{
						x:Neighbours{
							prev:self.get_wrap(pos+v3i(-1,0,0)),
							next:self.get_wrap(pos+v3i(1,0,0))},
						y:Neighbours{
							prev:self.get_wrap(pos+v3i(0,-1,0)),
							next:self.get_wrap(pos+v3i(0,1,0))},
						z:Neighbours{
							prev:self.get_wrap(pos+v3i(0,0,-1)),
							next:self.get_wrap(pos+v3i(0,0,1))}})
		})
	}
	/// special case of convolution for 2x2 cells, e.g. for marching cubes
	fn convolute_2x2x2_wrap<F,B>(&self,f:F)->Array3d<B>
		where F:Fn(V3i,[[[&Self::Output;2];2];2])->B,B:Clone
	{
		self.map_region(v3izero()..self.index_size(),|pos,_|{
			f(pos,[
				[	[self.get_ofs_wrap(pos,0, 0, 0),self.get_ofs_wrap(pos, 1, 0, 0)],
					[self.get_ofs_wrap(pos,0, 1, 0),self.get_ofs_wrap(pos, 1, 1, 0)]
				],
				[	[self.get_ofs_wrap(pos,0, 0, 1),self.get_ofs_wrap(pos, 1, 0, 1)],
					[self.get_ofs_wrap(pos,0, 1, 1),self.get_ofs_wrap(pos, 1, 1, 1)]
				]
			])
		})
	}

	/// take 2x2x2 blocks, fold to produce new values
	/// a b c d  -> f([[a,b],[e,f]]) f([[c,d],[g,h]])
	/// e f g h 
	fn fold_half_xyz<F,B>(&self,fold_fn:F)->Array3d<B>
		where F:Fn(V3i,[[[&Self::Output;2];2];2])->B,B:Clone
	{
		Array3d::from_fn( self.index_size()/v3i(2,2,2), |dpos:V3i|{
			let spos=dpos*v3i(2,2,2);
			fold_fn(dpos, self.get2x2x2(spos))
		})
	}
}


/*
fn avr<Diff,T>(a:&T,b:&T)->T where
	for<'u> Diff:Mul<f32,Output=Diff>+'u,
	for<'u,'v>&'u T:Sub<&'v T,Output=Diff>,
	for<'u,'v>&'u T:Add<&'v Diff,Output=T>
{
	a.add(&a.sub(b).mul(0.5f32))
}
*/
// for types T with arithmetic,
impl<T:Clone+Lerp> Array3d<T>
{
	/// downsample, TODO downsample centred on alternate cells
	fn downsample_half(&self)->Array3d<T>{
		self.fold_half_xyz(|pos:V3i,cell:[[[&T;2];2];2]|->T{
			
			avr(
				&avr(
					&avr(&cell[0][0][0],&cell[1][0][0]),
					&avr(&cell[0][1][0],&cell[1][1][0])),
				&avr(
					&avr(&cell[0][0][1],&cell[1][0][1]),
					&avr(&cell[0][1][1],&cell[1][1][1])
				)
			)

		})
	}

	/// expand 2x with simple interpolation (TODO , decent filters)
	fn upsample_double_axis(&self,axis:i32)->Array3d<T>{
		Array3d::from_fn_doubled(self.shape,axis,|pos:V3i|->(T,T){
			let v0=self.index(pos); let v1=self.get_wrap(v3iadd_axis(pos,axis,1));
			let vm=avr(v0,v1);
			(v0.clone(),vm)
		})
	}
	/// expand 2x in all axes (TODO, in one step instead of x,y,z composed)
	fn upsample_double_xyz(&self)->Array3d<T>{
		// todo - should be possible in one step, without 3 seperate buffer traversals!
		self.upsample_double_axis(XAxis).upsample_double_axis(YAxis).upsample_double_axis(ZAxis)
	}
}


impl<T> Index<V3i> for Array3d<T>{
	type Output=T;
	fn index(&self, pos:V3i)->&T{
		let i=self.linear_index(pos);
		&self.data[i]
	}
}

impl<T> IndexMut<V3i> for Array3d<T>{
	fn index_mut(&mut self, pos:V3i)->&mut T{
		let i=self.linear_index(pos);
		&mut self.data[i]
 	}
}

// implementing numeric operators
macro_rules! impl_binary_operators{[$(($traitname:ident,$opname:ident)),*]=>{
$(

	impl<'a,'b,A:Copy,B:Copy,R> $traitname<&'b Array3d<B>> for &'a Array3d<A> where A:$traitname<&'b B,Output=R>+'a, R:Clone{
		type Output=Array3d<R >;
		fn $opname(self,rhs:&'b Array3d<B>)->Self::Output{
			// todo -implement using 'zip_with', then specialize 'InternalIterators' for LinearIndexed
			assert!(self.index_size()==rhs.index_size());
			// todo - look into paired iterators 
			let mut out=Vec::new(); out.reserve(self.data.len());
			for x in 0..self.data.len(){
				out.push(self.at_linear_index(x).$opname(rhs.at_linear_index(x)))
			}
			Array3d::from_vec(self.index_size(),out)
		}
	}
)*
}}

//[$(($fname:ident,$fname_s:ident=>$op:ident)),*]
macro_rules! impl_assign_operators{[$(($traitname:ident,$opname:ident)),*]=>{$(

	impl<'a,'b,'d,'e,'f,'g,'arb, A,B:Copy> $traitname<&'arb Array3d<B>> for Array3d<A> where A:$traitname<B>,B:'e, Self:'f, &'g B:'g{
		fn $opname(&mut self,rhs:&'arb Array3d<B>){
			assert!(self.index_size()==rhs.index_size());
			// todo - look into paired iterators 
			for x in 0..self.data.len(){
				let rhsd=rhs.data.index(x);
				self.data[x].$opname(*rhsd);
			}
		}
	}

)*}}

impl_assign_operators!{(AddAssign,add_assign),(SubAssign,sub_assign),(MulAssign,mul_assign),(DivAssign,div_assign)}

impl_binary_operators!{(Add,add),(Sub,sub),(Mul,mul),(Div,div),(Shl,shl),(Shr,shr),(Rem,rem),(BitAnd,bitand),(BitXor,bitxor),(BitOr,bitor)}

//partialeq doesn't fit the pattern, must impl manually
impl<'a,'b,'c,'d,A,B> PartialEq<Array3d<B>> for Array3d<A> where A:PartialEq<B>+'a,B:'b{
	fn eq(&self,other:&Array3d<B>)->bool{
		
		assert!(self.index_size()==other.index_size());
		for x in 0..self.data.len(){
			if self.at_linear_index(x)!=other.at_linear_index(x){
				return false;
			}
		}
		return true;
	}
}
