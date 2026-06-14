import requests
from django.shortcuts import render, redirect, get_object_or_404
from .models import *
from django.db.models import Count
# Model Forms.
from .forms import PostAdsForm
from django.contrib.auth.forms import User
from django.contrib.auth.models import User

from django.contrib.auth.decorators import login_required

from django.conf import settings
from django.core.mail import send_mail

from django.utils import timezone
from datetime import timedelta
from django.db.models import Avg

# importing messages
from django.contrib import messages

from ads.models import Author
# Create your views here.

################# get IP
def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
###############

# Post ads view
@login_required(login_url='login')
def post_ads(request):
    if request.method == 'POST':
        # Get ad title
        title = request.POST.get('title')

        # Get ad description
        description = request.POST.get('description')

        # Get ad category
        category = request.POST.get('category')
        # Check if the category exists
        category_check = Category.objects.filter(category_name=category).exists()
        if category_check:
            c = Category.objects.get(category_name=category) # Get the category if exists
        else:
            c = Category.objects.create(category_name=category) # Create the category
        
        # Get ad price
        price = request.POST.get('price')
        
        # Get ad condition
        condition = request.POST.get('condition')
        
        # Get user's living state
        state = request.POST.get('state')
        # Check if the state exists
        state_check = State.objects.filter(state_name=state).exists()
        if state_check:
            s = State.objects.get(state_name=state) # Get the state if exists
        else:
            s = State.objects.create(state_name=state) # Create the state

        # Get user's living city
        city = request.POST.get('city')
        # Check if the city exists
        city_check = City.objects.filter(city_name=city).exists()
        if city_check:
            ci = City.objects.get(city_name=city) # Get the city if exists
        else:
            ci = City.objects.create(city_name=city) # Create the city

        # Get ad brand
        brand = request.POST.get('brand')

        # Get user's phone
        phone = request.POST.get('phone')

        # Get ad video
        video = request.POST.get('video')

        # Get image files length
        length = request.POST.get('length')

        # Create the ad
        ads = Ads.objects.create(
                                 author=request.user.author,
                                 title=title,
                                 description=description,
                                 price=price, category=c,
                                 condition=condition,
                                 state=s, city=ci,
                                 brand=brand,
                                 phone=phone,
                                 video=video
        )

        # Attach the images with the associated ad
        for file_num in range(0, int(length)):
            AdsImages.objects.create(
                ads=ads,
                image=request.FILES.get(f'images{file_num}')
            )

        ####################################################### ТУТ НАЧИНАЕТСЯ БЛОК ДЛЯ ДЖЕЙСОНА МОДЕЛЬКЕ ###########################################

        now = timezone.now()

        account_age = (now - request.user.date_joined).days

        ads_24h = Ads.objects.filter(author=request.user.author, date_created__gte=now - timedelta(days=1)).count()
        ads_7d = Ads.objects.filter(author=request.user.author, date_created__gte=now - timedelta(days=7)).count()
        rejected_count = Ads.objects.filter(author=request.user.author, is_active=False).count()
        
        is_verified = False

        if request.user.is_active:
            if hasattr(request.user.author, 'phone') and request.user.author.phone:
                is_verified = True

        full_text = f"{title} {description}".lower()

        # Проверка на Телеграм
        has_telegram = 'telegram' in full_text or ' t.me' in full_text or '@' in full_text

        # Проверка на срочность
        urgency_words = ['urgent', 'срочно', 'быстро', 'fast', 'sale', 'скидка']
        has_urgency = any(word in full_text for word in urgency_words)

        # Проверка на внешние контакты (ссылки, почта)
        external_words = ['http', 'www', '.com', '.ru', 'whatsapp', 'viber', 'пишите в']
        has_external = any(word in full_text for word in external_words)

        # Расчет соотношения цены со средней в этой категории
        avg_price = Ads.objects.filter(category=c).aggregate(Avg('price'))['price__avg']
        
        current_price = float(price) if price else 0.0
        if avg_price and avg_price > 0:
            ratio = current_price / float(avg_price)
        else:
            ratio = 1.0 # Если это первое объявление в категории

        # Forming JSON for AI model
        data_to_send = {
            "listing_id": str(ads.id),
            "title": str(title),
            "description": str(description),
            "price": current_price,
            "category": str(c.category_name),
            "location": f"{s.state_name}, {ci.city_name}",
            "account_age_days": int(account_age),
            "num_ads_last_24h": int(ads_24h),
            "num_ads_last_7d": int(ads_7d),
            "is_verified_user": is_verified,
            "previous_rejected_ads_count": int(rejected_count),
            "num_images": int(length),
            "has_telegram": bool(has_telegram),
            "has_urgency_word": bool(has_urgency),
            "has_external_contact": bool(has_external),
            "price_to_category_median_ratio": float(ratio)
        }

        ####################################################### ТУТ НАЧИНАЕТСЯ БЛОК ДЛЯ ОБЩЕНИЯ С МОДЕЛЬКОЙ ###########################################

        # URL нейросети по какому-то адресу. Руками поставил порт 8001 (по умолчанию ставится 8000)
        ML_SERVICE_URL = "http://127.0.0.1:8001/score" 

        ml_report = "Проверка нейросетью не была выполнена."
        level = "NEW"

        try:
            # Отправляем запрос
            response = requests.post(ML_SERVICE_URL, json=data_to_send, timeout=5)
            
            if response.status_code == 200:
                res = response.json() # получаем ответ (JSON)
                score = res.get('risk_score', 0)
                level = res.get('risk_level', 'unknown')
                factors = ", ".join(res.get('risk_factors', []))
                action = res.get('recommended_action', 'none')
                
                ml_report = (
                    f"Уровень риска: {level.upper()} ({score})\n"
                    f"Рекомендация: {action}\n"
                    f"Факторы риска: {factors if factors else 'не обнаружены'}\n"
                    f"Версия модели: {res.get('model_version')}"
                )

                fraud_alert = ""

                # Логика на основе ответа:
                if level == 'high': # сносим объявление сразу, если оно подозрительное
                    ads.is_active = False 
                    ads.save()
                    fraud_alert = f"ВНИМАНИЕ: Нейросеть оценила риск как ВЫСОКИЙ ({score})"
                else:
                    fraud_alert = "Нейросеть: Риск низкий."
            else:
                fraud_alert = "Нейросеть не ответила (ошибка сервера)."
        
        except Exception as e:
            ml_report = f"Ошибка при обращении к нейросети: {str(e)}"

        ####################################################### ОТПРАВЛЯЕМ ПИСЬМО АДМИНУ С ВЕРДИКТОМ ###########################################
        mail_subject = "New Ads submitted"

        message = (
            f"Админ, привет!\n\n"
            f"Пользователь {request.user.email} создал объявление: '{title}'\n"
            f"{fraud_alert}\n"
            f"--- ОТЧЕТ НЕЙРОСЕТИ ---\n"
            f"{ml_report}\n"
            f"------------------------\n\n"
            f"Посмотреть объявление в админке: http://127.0.0.1:8000/admin/ads/ads/{ads.id}/change/"
        )
        print(message)
        to_email = settings.EMAIL_HOST_USER
        to_list = [to_email]
        from_email = settings.EMAIL_HOST_USER
        
        send_mail(
            mail_subject,
            message,
            from_email,
            to_list,
            fail_silently=False,
        )
        
    return render(request, 'ads/post-ads.html')

# Ads listing view
def ads_listing(request):
    ads_listing = Ads.objects.all()
    category_listing = Category.objects.annotate(total_ads=Count('ads')).order_by('category_name')

    context = {
        'ads_listing' : ads_listing,
        'category_listing' : category_listing
    }

    return render(request, 'ads/ads-listing.html', context)

# Ads detail view
def ads_detail(request, pk):
    ads_detail = get_object_or_404(Ads, pk=pk)
    ads_photos = AdsImages.objects.filter(ads=ads_detail)

    context = {
        'ads_detail' : ads_detail,
        'ads_photos' : ads_photos,
    }

    return render(request, 'ads/ads-detail.html', context)

# Ads category archive view
def ads_category_archive(request, slug):
    category = get_object_or_404(Category, slug=slug)
    ads_by_category = Ads.objects.filter(category=category)

    context = {
        'category' : category,
        'ads_by_category' : ads_by_category
    }

    return render(request, 'ads/category-archive.html', context)

# Ads state archive view
def ads_state_archive(request, slug):
    state = get_object_or_404(State, slug=slug)
    ads_by_state = Ads.objects.filter(state=state)

    context = {
        'state' : state,
        'ads_by_state' : ads_by_state
    }

    return render(request, 'ads/state-archive.html', context)

# Ads city archive view
def ads_city_archive(request, slug):
    city = get_object_or_404(City, slug=slug)
    ads_by_city = Ads.objects.filter(city=city)

    context = {
        'city' : city,
        'ads_by_city' : ads_by_city
    }

    return render(request, 'ads/city-archive.html', context)

# Ads author archive view
def ads_author_archive(request, pk):
    author = get_object_or_404(Author, pk=pk)
    ads_by_author = Ads.objects.filter(author=author)

    context = {
        'author' : author,
        'ads_by_author' : ads_by_author
    }

    return render(request, 'ads/author-archive.html', context)

# Ads search/filter view
def ads_search(request):

    state = request.GET.get('state_name')
    category = request.GET.get('category_name')

    if state:
        ads_search_result = Ads.objects.filter(state__state_name=state)
    elif category:
        ads_search_result = Ads.objects.filter(category__category_name=category)
    else:
        ads_search_result = Ads.objects.filter(state__state_name=state).filter(category__category_name=category)
    
    context = {
        'ads_search_result':ads_search_result
    }

    return render(request, 'ads/ads-search.html', context)

# Ads delete view
@login_required(login_url='login')
def ads_delete(request, pk):
    ad = get_object_or_404(Ads, pk=pk)
    ad.delete()
    return redirect("dashboard")
