#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)] 
#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(slice_get_slice)]
#![feature(associated_type_defaults)]
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

}

pub mod tiled;
pub use tiled::*;
pub type Idx=i32;
use std::ops::Range;
use std::ops::{Add,Sub,Mul,Div,Rem,BitOr,BitAnd,BitXor,Index,IndexMut};
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

pub use vec_xyzw::{Vec2,Vec3,Vec4};
//pub struct Vec4<X,Y=X,Z=Y,W=Z>{pub x:X,pub y:Y,pub z:Z,pub w:W} // array3d::Vec3 should 'into()' into vector::Vec3 , etc.
pub type V2i=Vec2<Idx>;
pub type V3i=Vec3<Idx>;
pub type V4i=Vec4<Idx>;//TODO
type Axis_t=i32;

trait GetMut<I>{
	type Output;
	fn get_mut(&mut self, i:I)->&mut Self::Output;
}

impl<T:Copy> GetMut<i32> for Vec3<T>{
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
#[derive(Debug,Copy,Clone)]
pub struct Neighbours<T>{pub prev:T,pub next:T}
type Neighbours2d<T>=Vec2<Neighbours<T>>;
type Neighbours3d<T>=Vec3<Neighbours<T>>;
type Neighbours4d<T>=Vec4<Neighbours<T>>;

macro_rules! v3i_operators{[$(($fname:ident=>$op:ident)),*]=>{
	$(pub fn $fname(a:V3i,b:V3i)->V3i{v3i(a.x.$op(b.x),a.y.$op(b.y),a.z.$op(b.z))})*
}}
macro_rules! v3i_permute_v2i{[$($pname:ident($u:ident,$v:ident)),*]=>{
	$(pub fn $pname(a:V3i)->V2i{v2i(a.$u,a.$v)})*
}}

pub trait MyMod :Add+Sub+Div+Mul+Sized{
	fn mymod(&self,b:Self)->Self;
}
impl MyMod for i32{
	fn mymod(&self,b:Self)->Self{ if *self>=0{*self%b}else{ b-((-*self) %b)} }
}
v3i_operators![(v3iadd=>add),(v3isub=>sub),(v3imul=>mul),(v3idiv=>div),(v3irem=>rem),(v3imin=>min),(v3imax=>max),(v3imymod=>mymod)];
v3i_permute_v2i![v3i_xy(x,y), v3i_yz(y,z), v3i_xz(x,z)];

pub fn v3imuls(a:V3i,s:i32)->V3i{v3i(a.x*s,a.y*s,a.z*s)}
pub fn v3idivs(a:V3i,s:i32)->V3i{v3i(a.x/s,a.y/s,a.z/s)}
pub fn v3iands(a:V3i,s:i32)->V3i{v3i(a.x&s,a.y&s,a.z&s)}
pub fn v3ishrs(a:V3i,s:i32)->V3i{v3i(a.x>>s,a.y>>s,a.z>>s)}
pub fn v3ishls(a:V3i,s:i32)->V3i{v3i(a.x<<s,a.y<<s,a.z<<s)}
pub fn v3itilepos(a:V3i,tile_shift:i32)->(V3i,V3i){
	(v3ishrs(a,tile_shift),v3iands(a,(1<<tile_shift)-1))
}

pub fn v3i_hmul_usize(a:V3i)->usize{ a.x as usize*a.y as usize *a.z as usize}
pub fn v2i_hmul_usize(a:V2i)->usize{ a.x as usize*a.y as usize}
pub fn v2i_hadd_usize(a:V2i)->usize{ a.x as usize+a.y as usize}
pub fn v3i_hadd_usize(a:V3i)->usize{ a.x as usize+a.y as usize +a.z as usize}

pub struct Array2d<T>{pub shape:V2i,pub data:Vec<T>}
pub struct Array3d<T>{pub shape:V3i,pub data:Vec<T>}
pub struct Array4d<T>{pub shape:V4i,pub data:Vec<T>}

fn eval_linear_index(pos:V3i, stride:V3i)->usize{
	(pos.x as usize)*(stride.x as usize)+(pos.y as usize) * (stride.y as usize) + (pos.z as usize)*(stride.z as usize)
}


/// utility trait: types which can convert XYZ<->linear indices
pub trait XYZIndexable : IndexMut<V3i>{
	fn index_size(&self)->V3i;
}
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
/// Iterator over 3d index positions and associated linear index computed from strides
pub struct IterXYZ {
	state:IterXYZState,
	range:IterXYZRange
}
pub type IterXYZItem=(V3i,usize);
fn step_xyz_iter(s:&mut IterXYZState,range:&IterXYZRange)->Option<IterXYZItem>{
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

/// region of array3d, analogous to array slices.
/// 'TS'=collecton of T
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
	/// consume a vec to build
	fn from_vec(size:V3i,v:Vec<T>)->Self{
		assert!((size.x*size.y*size.z) as usize==v.len());
		Array3d{shape:size,data:v}
	}
	/// destructure
	fn to_vec(self)->(V3i,Vec<T>){
		(self.shape,self.data)
	}
	/// get a region, TODO can this be made to 
	/// work with rust's nifty array syntax?
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
	fn iter_cells<'a>(&'a self)->Array3dIter<'a,T>{
		Array3dIter(self, IterXYZ::new(v3i(0,0,0),self.shape, 0 , self.linear_stride()))
	}
	fn iter_cells_mut<'a>(&'a mut self)->Array3dIterMut<'a,T>{
		let shp=self.shape.clone();
		let strd=self.linear_stride();
		Array3dIterMut(self, IterXYZ::new(v3i(0,0,0),shp, 0 , strd))
	}
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

impl<'a,T:Clone> IterXYZAble<'a> for Array3d<T>{
	type Elem=T;
	fn at_linear_index(&'a self,i:LinearIndex)->&'a T{ self.data.index(i)}
}
impl<'a,T:Clone> IterXYZAbleMut<'a> for Array3d<T>{
	fn at_linear_index_mut(&'a mut self,i:LinearIndex)->&'a mut T{ self.data.index_mut(i)}
}

impl<T:Clone> Array3d<T>{
	
	pub fn from_fn<F:Fn(V3i)->T> (s:V3i,f:F) -> Array3d<T> {
		let mut a=Array3d{shape:s, data:Vec::new()};
		a.data.reserve(v3i_hmul_usize(s));
		for z in 0..a.shape.z{ for y in 0..a.shape.y { for x in 0.. a.shape.x{
			a.data.push( f(v3i(x,y,z)) )
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
	/// produce a 2d array by folding along the X axis
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


/// internal iterators
/// until rust has HKT/ATOC, options for output are limited,
/// we keep flat Array3d as the default output
/// Default implementations just built on indexing, however addressgen would
/// be inefficient this way
/// implementation for specific arrangements could implement certain
/// traversals more efficiently (be it flat, morton order, whatever)
pub trait XYZInternalIterators : XYZIndexable{
	type T=Self::Output;

	fn map_strided_region<B:Clone,F:Fn(V3i,&Self::Output)->B> 
		(&self,range:Range<V3i>,stride:V3i, f:F) -> Array3d<B>
	{
		Array3d::from_fn(v3idiv(v3isub(range.end,range.start),stride),
			|outpos:V3i|{
				let inpos=v3iadd(v3imul(outpos,stride),range.start);
				f(inpos,self.index(inpos))
			}
		)
	}

	/// internal iteration with inplace mutation
	fn map_xyz<B:Clone,F:Fn(V3i,&Self::Output)->B> (&self,f:F) -> Array3d<B>{
		Array3d::from_fn(self.index_size(),
			|pos:V3i|f(pos,self.index(pos))
		)
	}
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
	fold_axis!{fn fold_x traverse (y,z) reduce x}
	fold_axis!{fn fold_y traverse (x,z) reduce y}
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
	/// fold values along x,y,z in turn without intermediate storage
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
	/// fold values along z,y,x in turn without intermediate storage
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
	/// output size is divided by tilesize
	/// must be exact multiple.
	fn fold_tiles<B,F>(&self,tilesize:V3i, input:B,f:&F)->Array3d<B>
		where F:Fn(V3i,B,&Self::Output)->B,B:Clone
	{
		self.map_strided(tilesize,
			|pos,_:&Self::Output|{self.fold_region(pos..v3iadd(pos,tilesize),input.clone(),f)})
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
		Array3d::from_fn(v3idiv(v3isub(region.end,region.start),stride),
			|outpos:V3i|{
				let inpos=v3iadd(region.start,v3imul(outpos,stride));
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
		self.map_region_strided(region, v3ione(), f)
	}
	/// _X_     form of convolution  
	/// XOX		passing each cell and it's
	/// _X_		immiediate neighbours on each axis
	fn convolute_neighbours<F,B>(&self,f:F)->Array3d<B>
		where F:Fn(&Self::Output,Vec3<Neighbours<&Self::Output>>)->B ,B:Clone
	{
		self.map_region(v3ione()..v3isub(self.index_size(),v3ione()),
			|pos:V3i,current_cell:&Self::Output|{
				f(	current_cell,
					self::Vec3{
						x:Neighbours{
							prev:self.index(v3i(pos.x-1,pos.y,pos.z)),
							next:self.index(v3i(pos.x+1,pos.y,pos.z))},
						y:Neighbours{
							prev:self.index(v3i(pos.x,pos.y-1,pos.z)),
							next:self.index(v3i(pos.x,pos.y+1,pos.z))},
						z:Neighbours{
							prev:self.index(v3i(pos.x,pos.y,pos.z-1)),
							next:self.index(v3i(pos.x,pos.y,pos.z+1))}})
		})
	}
	fn index_wrap(&self,pos:V3i)->&Self::Output{self.get_wrap(pos)}
	fn get_wrap(&self,pos:V3i)->&Self::Output{
		self.index( v3imymod(pos, self.index_size()) )
	}
	fn get_ofs_wrap(&self,pos:V3i,dx:i32,dy:i32,dz:i32)->&Self::Output{
		self.get_wrap(v3iadd(pos, v3i(dx,dy,dz)))
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
							prev:self.get_wrap(v3i(pos.x-1,pos.y,pos.z)),
							next:self.get_wrap(v3i(pos.x+1,pos.y,pos.z))},
						y:Neighbours{
							prev:self.get_wrap(v3i(pos.x,pos.y-1,pos.z)),
							next:self.get_wrap(v3i(pos.x,pos.y+1,pos.z))},
						z:Neighbours{
							prev:self.get_wrap(v3i(pos.x,pos.y,pos.z-1)),
							next:self.get_wrap(v3i(pos.x,pos.y,pos.z+1))}})
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
		Array3d::from_fn( v3idiv(self.index_size(),v3i(2,2,2)), |dpos:V3i|{
			let spos=v3imul(dpos,v3i(2,2,2));
			fold_fn(dpos,
					[	[	[self.index(v3iadd(spos,v3i(0,0,0))),self.index(v3iadd(spos,v3i(1,0,0)))],
							[self.index(v3iadd(spos,v3i(0,1,0))),self.index(v3iadd(spos,v3i(1,1,0)))]
						],
						[	[self.index(v3iadd(spos,v3i(0,0,0))),self.index(v3iadd(spos,v3i(1,0,0)))],
							[self.index(v3iadd(spos,v3i(0,0,0))),self.index(v3iadd(spos,v3i(1,0,0)))]
						]
					]
			)
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






