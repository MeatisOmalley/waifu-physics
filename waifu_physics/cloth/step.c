/* Waifu Cloth float32 PBD hot loop. Independently authored behavioral port;
 * no Epic source is incorporated. Numerical sources: docs/cloth-reference.md.
 * Build without fast math/FMA contraction for reproducible scalar arithmetic. */
#include <math.h>
#include <stdint.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    int n, ng;
    float dt;
    float *x, *p, *v, *target, *anim_vel, *inv_mass, *max_distance;
    float *anim_stiff, *anim_damp, *acceleration, *local_dv;
    int32_t *starts, *iterations;
    float *params;
    int32_t *edges;
    float *edge_length;
    int32_t *edge_starts, *bending;
    float *bending_length;
    int32_t *bending_starts, *area;
    float *area_length, *area_bary;
    int32_t *area_starts, *tethers;
    float *tether_length;
    int32_t *tether_starts;
    float *shapes;
    int32_t *shape_starts;
} ClothState;

static float minimum(float a, float b) { return a < b ? a : b; }
static float maximum(float a, float b) { return a > b ? a : b; }
static float dot(const float *a, const float *b) { return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]; }
static float length(const float *a) { return sqrtf(dot(a,a)); }
static void cross(const float *a, const float *b, float *out) {
    out[0]=a[1]*b[2]-a[2]*b[1]; out[1]=a[2]*b[0]-a[0]*b[2]; out[2]=a[0]*b[1]-a[1]*b[0];
}

static void springs(ClothState *s, int32_t *pairs, float *rest, int first, int last, float stiffness) {
    int k,j;
    if (stiffness == 0) return;
    for (k=first;k<last;++k) {
        int a=pairs[2*k], b=pairs[2*k+1];
        float d[3], dist, w=s->inv_mass[b]+s->inv_mass[a], sq;
        for (j=0;j<3;++j) d[j]=s->p[3*a+j]-s->p[3*b+j];
        sq=dot(d,d); dist=sqrtf(sq);
        if (w <= 0) continue;
        if (sq < 1e-4f) { d[0]=1; d[1]=d[2]=0; dist=0; }
        else for(j=0;j<3;++j) d[j]/=dist;
        for (j=0;j<3;++j) {
            float inner=(dist-rest[k])*d[j];
            float delta=(stiffness*inner)/w;
            s->p[3*a+j]-=s->inv_mass[a]*delta;
            s->p[3*b+j]+=s->inv_mass[b]*delta;
        }
    }
}

static void axial(ClothState *s, int first, int last, float stiffness) {
    int k,j;
    if (stiffness == 0) return;
    for (k=first;k<last;++k) {
        int a=s->area[3*k], b=s->area[3*k+1], c=s->area[3*k+2];
        float bary=s->area_bary[k], other=1.f-bary;
        float d[3], dist, w=(s->inv_mass[c]*other+s->inv_mass[b]*bary)+s->inv_mass[a];
        float multiplier=2.f/(maximum(bary,other)+1.f);
        for (j=0;j<3;++j) d[j]=s->p[3*a+j]-((s->p[3*b+j]-s->p[3*c+j])*bary+s->p[3*c+j]);
        dist=length(d);
        if (dist <= 1e-8f || w <= 0) continue;
        for (j=0;j<3;++j) {
            float direction=d[j]/dist;
            float inner=(dist-s->area_length[k])*direction;
            float delta=(stiffness*inner)/w;
            s->p[3*a+j]-=(multiplier*s->inv_mass[a])*delta;
            s->p[3*b+j]+=((multiplier*s->inv_mass[b])*bary)*delta;
            s->p[3*c+j]+=((multiplier*s->inv_mass[c])*other)*delta;
        }
    }
}

static void collide(ClothState *s, int g) {
    int i,k,j,t;
    float thick=s->params[8*g+6], friction=s->params[8*g+7];
    for (k=s->shape_starts[g];k<s->shape_starts[g+1];++k) {
        float *shape=s->shapes+25*k, *r=shape+4;
        int kind=(int)shape[0];
        for (i=s->starts[g];i<s->starts[g+1];++i) {
            float local[3], offset[3], normal[3]={0,0,0}, worldnormal[3];
            float phi, penetration, dist;
            if (s->inv_mass[i] == 0) continue;
            for (j=0;j<3;++j) offset[j]=s->p[3*i+j]-shape[1+j];
            for (j=0;j<3;++j) local[j]=(offset[0]*r[j]+offset[1]*r[3+j])+offset[2]*r[6+j];
            if (kind==0 || kind==2 || kind==3) {
                float d[3]={local[0],local[1],local[2]}, radius=shape[13];
                if (kind==2 || kind==3) {
                    float z=minimum(maximum(local[2],-shape[15]),shape[15]);
                    d[2]-=z;
                    if (kind==3) radius=shape[15]>1e-8f ? shape[14]+((shape[13]-shape[14])*(z+shape[15]))/(2.f*shape[15]) : maximum(shape[13],shape[14]);
                }
                dist=length(d);
                if (dist>1e-8f) for (j=0;j<3;++j) normal[j]=d[j]/dist;
                else normal[0]=1;
                phi=dist-radius;
            } else if (kind==4) {
                float q[3], outside[3], maxq;
                int axis=0;
                for (j=0;j<3;++j) { q[j]=fabsf(local[j])-shape[16+j]; outside[j]=maximum(q[j],0); }
                dist=length(outside); maxq=q[0];
                for (j=1;j<3;++j) if(q[j]>maxq) { maxq=q[j]; axis=j; }
                if(dist>1e-8f) for(j=0;j<3;++j) normal[j]=outside[j]*(local[j]<0 ? -1.f:1.f)/dist;
                else normal[axis]=local[axis]<0 ? -1.f:1.f;
                phi=dist+minimum(maxq,0);
            } else { normal[2]=1; phi=local[2]; }
            penetration=thick-phi;
            if (penetration<=0) continue;
            for(j=0;j<3;++j) worldnormal[j]=(normal[0]*r[3*j]+normal[1]*r[3*j+1])+normal[2]*r[3*j+2];
            for(j=0;j<3;++j) s->p[3*i+j]+=penetration*worldnormal[j];
            if (friction>1e-4f) {
                float radial[3], angular[3], displacement[3], tangent[3], along, ratio;
                for(j=0;j<3;++j) radial[j]=s->p[3*i+j]-shape[1+j];
                cross(shape+22,radial,angular);
                for(j=0;j<3;++j) displacement[j]=(s->p[3*i+j]-s->x[3*i+j])-(shape[19+j]+angular[j])*s->dt;
                along=dot(displacement,worldnormal);
                for(j=0;j<3;++j) tangent[j]=displacement[j]-worldnormal[j]*along;
                dist=length(tangent);
                ratio=minimum(penetration*friction,dist)/maximum(dist,1e-8f);
                for(j=0;j<3;++j) s->p[3*i+j]-=ratio*tangent[j];
            }
        }
    }
}

API int waifu_cloth_version(void) { return 1; }
API void waifu_cloth_step(ClothState *s) {
    int g,i,j,k,iteration;
    for(i=0;i<s->n;++i) for(j=0;j<3;++j) {
        if(s->inv_mass[i]>0) { s->v[3*i+j]+=s->acceleration[3*i+j]*s->dt; s->v[3*i+j]+=s->local_dv[3*i+j]; }
        s->p[3*i+j]=s->target[3*i+j];
    }
    for(g=0;g<s->ng;++g) {
        float *params=s->params+8*g;
        for(i=s->starts[g];i<s->starts[g+1];++i) if(s->inv_mass[i]>0)
            for(j=0;j<3;++j) s->p[3*i+j]=s->x[3*i+j]+s->v[3*i+j]*params[0];
        for(k=s->tether_starts[g];k<s->tether_starts[g+1];++k) {
            int a=s->tethers[2*k], b=s->tethers[2*k+1];
            float d[3], dist, offset, scale;
            for(j=0;j<3;++j) d[j]=s->p[3*a+j]-s->p[3*b+j];
            dist=length(d); offset=maximum(dist-s->tether_length[k]*params[5],0);
            scale=(params[4]*offset)/maximum(dist,1e-8f);
            for(j=0;j<3;++j) s->p[3*b+j]+=d[j]*scale;
        }
        for(iteration=0;iteration<s->iterations[g];++iteration) {
            springs(s,s->edges,s->edge_length,s->edge_starts[g],s->edge_starts[g+1],params[1]);
            springs(s,s->bending,s->bending_length,s->bending_starts[g],s->bending_starts[g+1],params[2]);
            axial(s,s->area_starts[g],s->area_starts[g+1],params[3]);
            for(i=s->starts[g];i<s->starts[g+1];++i) if(s->inv_mass[i]>0) {
                float d[3],dist;
                for(j=0;j<3;++j) d[j]=s->p[3*i+j]-s->target[3*i+j];
                dist=length(d);
                if(dist>s->max_distance[i]) for(j=0;j<3;++j) s->p[3*i+j]=s->target[3*i+j]+d[j]*(s->max_distance[i]/dist);
                for(j=0;j<3;++j) {
                    s->p[3*i+j]-=s->anim_stiff[i]*(s->p[3*i+j]-s->target[3*i+j]);
                    s->p[3*i+j]-=s->anim_damp[i]*((s->p[3*i+j]-s->x[3*i+j])-s->anim_vel[3*i+j]*s->dt);
                }
            }
            collide(s,g);
        }
    }
    for(i=0;i<s->n*3;++i) { s->v[i]=(s->p[i]-s->x[i])/s->dt; s->x[i]=s->p[i]; }
}
